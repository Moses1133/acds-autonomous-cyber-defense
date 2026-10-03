"""
CICIDS2017-calibrated attack generator.

Reads data/cicids_stats.json (extracted from 2.8M labeled flows) and
generates attacks whose damage/intensity distributions match those
observed in the real CICIDS2017 dataset.

Mapping from CICIDS flow features to env damage units:
  raw_damage ~ log10(Flow Bytes/s mean) / 10   [0..1]
  weight     ~ log10(n_flows_total) / 10       [0..1]
  variance   ~ std/mean of Flow Bytes/s        (multiplier on sampled damage)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os
import math

import numpy as np


# Attack group ids (must match env's ATTACK_* ids)
GROUP_DOS         = "dos"
GROUP_RECON       = "recon"
GROUP_BRUTE_FORCE = "brute_force"
GROUP_BOT         = "bot"
GROUP_WEB         = "web"      # not present in every dataset; safe fallback

ALL_GROUPS = [GROUP_DOS, GROUP_RECON, GROUP_BRUTE_FORCE, GROUP_BOT, GROUP_WEB]


@dataclass
class CICIDSProfile:
    """Calibrated attack profile for one attack group."""
    group: str
    raw_damage_mean: float      # in [0, 1]
    raw_damage_std:  float      # in [0, 1]
    weight:          float      # attack pick weight
    n_flows:         int        # for reporting


def _safe_log10(x: float) -> float:
    return math.log10(max(x, 1.0))


class CICIDSAttackGenerator:
    """Generates attacks calibrated to real CICIDS2017 statistics."""

    def __init__(self, stats_path: Optional[str] = None):
        if stats_path is None:
            root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            stats_path = os.path.join(root, "data", "cicids_stats.json")

        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"CICIDS stats not found at {stats_path}. "
                f"Run scripts\\extract_cicids_stats.py first."
            )

        with open(stats_path) as fh:
            self.stats = json.load(fh)

        self.profiles: Dict[str, CICIDSProfile] = {}
        self._build_profiles()

    # ---------------- profile construction ----------------

    def _build_profiles(self) -> None:
        groups = self.stats.get("groups", {})
        for group, data in groups.items():
            feats = data.get("features", {})
            bytes_stats = feats.get("Flow Bytes/s") or feats.get("Total Fwd Packets") or {}

            mean_bytes = bytes_stats.get("mean", 1.0)
            std_bytes  = bytes_stats.get("std", 0.0)
            n_flows    = data.get("n_flows_total", 1)

            # Map to [0, 1]:
            # log10(mean_bytes) is ~[0, 8] for realistic traffic
            raw_damage_mean = _safe_log10(mean_bytes) / 10.0
            raw_damage_std  = _safe_log10(std_bytes)  / 10.0 if std_bytes > 0 else 0.02
            weight          = _safe_log10(n_flows)    / 10.0

            self.profiles[group] = CICIDSProfile(
                group=group,
                raw_damage_mean=float(np.clip(raw_damage_mean, 0.02, 0.9)),
                raw_damage_std=float(np.clip(raw_damage_std, 0.01, 0.3)),
                weight=float(np.clip(weight, 0.1, 1.0)),
                n_flows=int(n_flows),
            )

        # Ensure every group has a profile (fallback for missing ones)
        for g in ALL_GROUPS:
            if g not in self.profiles:
                self.profiles[g] = CICIDSProfile(
                    group=g,
                    raw_damage_mean=0.15,
                    raw_damage_std=0.05,
                    weight=0.5,
                    n_flows=0,
                )

    # ---------------- sampling ----------------

    def sample_damage(self, group: str, rng: np.random.Generator) -> float:
        """Sample a raw damage value for the given group, calibrated to
        real CICIDS2017 distributions."""
        profile = self.profiles.get(group)
        if profile is None:
            return float(rng.uniform(0.05, 0.15))

        # Gaussian-ish around mean with std from data, clipped to [0.02, 0.9]
        raw = float(rng.normal(profile.raw_damage_mean, profile.raw_damage_std))
        return float(np.clip(raw, 0.02, 0.9))

    def sample_group(self, stage: int, rng: np.random.Generator) -> str:
        """Choose which attack group fires this step, weighted by real
        CICIDS flow counts and constrained by the current kill-chain stage.

        stage: 1 = recon, 2 = exploit, 3 = escalate
        """
        # Stage-based candidate restriction
        if stage == 1:
            candidates = [GROUP_RECON, GROUP_DOS]
        elif stage == 2:
            candidates = [GROUP_BRUTE_FORCE, GROUP_DOS, GROUP_BOT]
        else:  # stage 3
            candidates = [GROUP_DOS, GROUP_BOT, GROUP_BRUTE_FORCE, GROUP_WEB]

        weights = np.array(
            [self.profiles[c].weight for c in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return str(rng.choice(candidates, p=weights))

    # ---------------- reporting ----------------

    def summary(self) -> str:
        lines = ["CICIDSAttackGenerator profiles:"]
        for g, p in sorted(self.profiles.items()):
            lines.append(
                f"  {g:14s} damage(mean±std)={p.raw_damage_mean:.3f}±{p.raw_damage_std:.3f}  "
                f"weight={p.weight:.3f}  n_flows={p.n_flows:,}"
            )
        return "\n".join(lines)

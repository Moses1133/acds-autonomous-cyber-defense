"""
CyberDefenseEnv v7 - v6 + CICIDS2017-calibrated attacks.

V7 CHANGES:
- Attack damage/weight distributions come from real CICIDS2017 statistics
  (via CICIDSAttackGenerator)
- Everything else (MultiDiscrete action space, kill chains, observation) is
  identical to v6.
"""

from __future__ import annotations
from typing import List
import numpy as np

from src.sim.environment_v6 import (
    CyberDefenseEnvV6, Attack,
    N_HOSTS, MAX_STEPS,
    N_ACTION_TYPES,
    STAGE_RECON, STAGE_EXPLOIT, STAGE_ESCALATE,
    ATTACK_NONE, ATTACK_DOS, ATTACK_BRUTE_FORCE, ATTACK_EXPLOIT,
    NUM_ATTACKS_PER_STEP_RANGE,
)

# Map CICIDS group name -> ATTACK_* id used in the env's damage model
GROUP_TO_ATTACK_ID = {
    "dos":         ATTACK_DOS,
    "recon":       ATTACK_DOS,          # recon uses DoS damage path
    "brute_force": ATTACK_BRUTE_FORCE,
    "bot":         ATTACK_EXPLOIT,
    "web":         ATTACK_EXPLOIT,
}


class CyberDefenseEnvV7(CyberDefenseEnvV6):
    """v6 + CICIDS-calibrated attacks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lazy-load CICIDS generator
        from src.attacks.cicids import CICIDSAttackGenerator
        self.cicids = CICIDSAttackGenerator()

        # Expose profiles for inspection / logging
        self.cicids_profiles = self.cicids.profiles

    def _generate_attacks(self):
        hosts = list(self.net.hosts.values())
        if not hosts:
            return []

        num_attacks = int(self._rng.integers(*NUM_ATTACKS_PER_STEP_RANGE))

        # Host attackability weighting
        weights = []
        for h in hosts:
            w = 1.0
            w += 1.5 * len(h.vulnerabilities)
            w += 0.5 * len(h.open_ports)
            if h.compromised:
                w *= 0.3
            weights.append(w)
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        chosen = self._rng.choice(len(hosts), size=num_attacks,
                                  replace=True, p=weights)

        attacks: List[Attack] = []
        for idx in chosen:
            h = hosts[int(idx)]

            # Sample attack group from CICIDS distribution (stage-aware)
            group = self.cicids.sample_group(self._stage, self._rng)
            atk_type = GROUP_TO_ATTACK_ID.get(group, ATTACK_DOS)

            # Sample raw damage from CICIDS-calibrated distribution
            raw = self.cicids.sample_damage(group, self._rng)

            # For recon stage, scale damage down (recon is stealthy)
            if self._stage == STAGE_RECON:
                raw *= 0.35
            # For escalate stage, scale up (final push)
            elif self._stage == STAGE_ESCALATE:
                raw *= 1.25

            raw = float(np.clip(raw, 0.02, 0.5))

            attacks.append(Attack(
                target_id=h.id,
                attack_type=atk_type,
                stage=self._stage,
                raw_damage=raw,
            ))

        return attacks

    def _get_info(self):
        info = super()._get_info()
        info["cicids_calibrated"] = True
        return info


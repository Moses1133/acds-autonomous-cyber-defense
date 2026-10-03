"""
CyberDefenseEnv v6 - extended environment with multi-discrete actions and
kill-chain attacks.

V6 CHANGES (real-world extension, Phase 1):
- Action space: MultiDiscrete([5, n_hosts]) -- agent picks (action_type, target_host)
- Structured attacks: kill chains (recon -> exploit -> escalate)
- Richer observation: last action, last target, attack stage, per-host alert flags
- Backward-compatible host vectors (Host.to_vector unchanged)

Backward compatibility: this file does not replace v5. Both can be loaded
side by side.

Action space:
    0 = noop           (target ignored)
    1 = patch          (target host, removes a CVE)
    2 = block_port     (target host, closes one port)
    3 = isolate        (target host, removes all ports, small health cost)
    4 = scan           (target host, reveals hidden state)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.sim.host import Host
from src.sim.network import Network


# ---------- configuration ----------

N_HOSTS = 10
MAX_STEPS = 100

# Action type ids (must stay stable)
ACTION_NOOP       = 0
ACTION_PATCH      = 1
ACTION_BLOCK_PORT = 2
ACTION_ISOLATE    = 3
ACTION_SCAN       = 4
N_ACTION_TYPES = 5

# Attack stages (kill chain)
STAGE_NONE     = 0
STAGE_RECON    = 1
STAGE_EXPLOIT  = 2
STAGE_ESCALATE = 3

# Attack type ids
ATTACK_NONE         = 0
ATTACK_DOS          = 1
ATTACK_BRUTE_FORCE  = 2
ATTACK_EXPLOIT      = 3

# Reward shaping (tuned from v5)
R_PER_HEALTHY_STEP    =  +0.04
R_PER_COMPROMISED     =  -0.45
R_DAMAGE_PREVENTED    =  +0.70
R_ACTION_COST         =  -0.04
R_TERMINAL_WIN        = +20.0
R_TERMINAL_LOSS       = -22.0

# Damage model
RECOVERY_PER_STEP = 0.02
NUM_ATTACKS_PER_STEP_RANGE = (1, 4)
WIN_THRESHOLD  = 7
LOSS_THRESHOLD = 3

# Observation size (host vectors + network summary + new fields)
HOST_VECTOR_SIZE = 7   # from Host.to_vector()
NETWORK_SUMMARY_SIZE = 4
RESERVED_SIZE = 2
LAST_ACTION_SIZE = N_ACTION_TYPES          # 5
LAST_TARGET_SIZE = N_HOSTS                 # 10
ATTACK_STAGE_SIZE = 3                      # one-hot over {recon, exploit, escalate}
ATTACK_ALERT_SIZE = N_HOSTS                # per-host "under attack" flag

CVE_POOL = [
    "CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003",
    "CVE-2024-0004", "CVE-2024-0005", "CVE-2024-0006",
]


def obs_size(n_hosts: int = N_HOSTS) -> int:
    return (
        n_hosts * HOST_VECTOR_SIZE
        + NETWORK_SUMMARY_SIZE
        + RESERVED_SIZE
        + LAST_ACTION_SIZE
        + LAST_TARGET_SIZE
        + ATTACK_STAGE_SIZE
        + ATTACK_ALERT_SIZE
    )


@dataclass
class Attack:
    target_id: int
    attack_type: int    # ATTACK_* id
    stage: int          # STAGE_* id
    raw_damage: float


class CyberDefenseEnvV6(gym.Env):
    """Extended cyber defense environment with kill-chain attacks and
    multi-discrete actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_hosts: int = N_HOSTS, max_steps: int = MAX_STEPS):
        super().__init__()
        self.n_hosts = n_hosts
        self.max_steps = max_steps
        self._step_count = 0

        self.net = self._build_network(n_hosts)

        # ---- spaces ----
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size(n_hosts),), dtype=np.float32
        )
        # Multi-discrete: [action_type, target_host]
        self.action_space = spaces.MultiDiscrete([N_ACTION_TYPES, n_hosts])

        self._rng = np.random.default_rng()
        self._pending_attacks: List[Attack] = []
        self._last_damage_events: List[Dict[str, Any]] = []

        # Kill chain state
        self._stage: int = STAGE_NONE
        self._stages_completed = 0

        # Observation extras
        self._last_action: int = -1
        self._last_target: int = -1
        self._under_attack: List[bool] = [False] * n_hosts

    # ---------------- construction ----------------

    def _build_network(self, n: int) -> Network:
        net = Network()
        net.add_host(Host(id=0, ip="10.0.0.1", role="router",
                          open_ports=[22, 53, 80]))
        for i in range(1, n):
            role = "server" if i % 3 == 0 else "workstation"
            ports = [22, 80, 443] if role == "server" else [22, 3389]
            net.add_host(Host(id=i, ip=f"10.0.0.{i+1}", role=role, open_ports=ports))
            net.add_link(0, i)
        return net

    def _assign_random_vulnerabilities(self) -> None:
        for h in self.net.hosts.values():
            h.vulnerabilities = []
        candidates = [h for h in self.net.hosts.values() if h.role != "router"]
        for h in candidates:
            n_cve = int(self._rng.integers(1, 3))
            h.vulnerabilities = list(
                self._rng.choice(CVE_POOL, size=n_cve, replace=False)
            )

    # ---------------- gym API ----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Rebuild the network to guarantee a fresh starting state
        # (Host.reset() preserves open_ports by design, which would leak
        # state between episodes if we reused the same Host objects)
        self.net = self._build_network(self.n_hosts)
        self._step_count = 0
        self._pending_attacks = []
        self._last_damage_events = []
        self._assign_random_vulnerabilities()

        self._stage = STAGE_RECON
        self._stages_completed = 0
        self._last_action = -1
        self._last_target = -1
        self._under_attack = [False] * self.n_hosts

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Multi-discrete action: [action_type, target_host]
        action = np.asarray(action).astype(int)
        action_type = int(action[0])
        target_host = int(action[1])

        self._step_count += 1
        self.net.current_step = self._step_count

        action_effect = self._apply_action(action_type, target_host)
        self._pending_attacks = self._generate_attacks()
        attack_effect = self._resolve_attacks()
        self._apply_recovery()

        reward = self._compute_reward(action_type, action_effect, attack_effect)

        terminated, win = self._check_termination()
        if terminated:
            reward += R_TERMINAL_WIN if win else R_TERMINAL_LOSS

        truncated = self._step_count >= self.max_steps
        if truncated:
            n_healthy = sum(1 for h in self.net.hosts.values() if not h.compromised)
            if n_healthy >= WIN_THRESHOLD:
                reward += R_TERMINAL_WIN
                terminated = True

        self._last_action = action_type
        self._last_target = target_host

        # Update kill-chain stage
        self._update_stage()

        obs = self._get_obs()
        info = self._get_info()
        info["action_effect"] = action_effect
        info["attack_effect"] = attack_effect
        info["stage"] = int(self._stage)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        comp = [h.id for h in self.net.compromised_hosts()]
        avg_h = np.mean([h.health for h in self.net.hosts.values()])
        print(f"[step {self._step_count:3d}] stage={self._stage} "
              f"compromised={comp} avg_health={avg_h:.2f}")

    # ---------------- action effects ----------------

    def _apply_action(self, action_type: int, target_id: int) -> Dict[str, Any]:
        effect: Dict[str, Any] = {
            "action_type": action_type,
            "target": target_id,
            "applied": False,
        }

        if action_type == ACTION_NOOP:
            effect["applied"] = True
            return effect

        # Validate target
        if target_id not in self.net.hosts:
            return effect
        target = self.net.hosts[target_id]

        if action_type == ACTION_PATCH:
            if target.vulnerabilities:
                cve = self._rng.choice(target.vulnerabilities)
                target.patch(cve)
                effect.update(applied=True, removed=cve)
            else:
                effect.update(reason="no_vulns")

        elif action_type == ACTION_BLOCK_PORT:
            if target.open_ports:
                port = int(self._rng.choice(target.open_ports))
                target.open_ports.remove(port)
                effect.update(applied=True, blocked_port=port)
            else:
                effect.update(reason="no_ports")

        elif action_type == ACTION_ISOLATE:
            removed = list(target.open_ports)
            target.open_ports = []
            target.take_damage(0.03)
            effect.update(applied=True, removed_ports=removed)

        elif action_type == ACTION_SCAN:
            effect.update(
                applied=True,
                revealed={
                    "health": float(target.health),
                    "compromised": bool(target.compromised),
                    "n_vulns": len(target.vulnerabilities),
                    "ports": list(target.open_ports),
                },
            )

        self.net.log_event("defense", **effect)
        return effect

    def _most_at_risk_host(self) -> Optional[Host]:
        candidates = [h for h in self.net.hosts.values() if h.role != "router"]
        if not candidates:
            return None
        return min(candidates, key=lambda h: (h.health, h.id))

    # ---------------- kill-chain attack generator ----------------

    def _update_stage(self) -> None:
        """Advance the kill chain based on elapsed steps."""
        t = self._step_count
        if t < 30:
            self._stage = STAGE_RECON
        elif t < 70:
            self._stage = STAGE_EXPLOIT
        else:
            self._stage = STAGE_ESCALATE

    def _generate_attacks(self) -> List[Attack]:
        hosts = list(self.net.hosts.values())
        if not hosts:
            return []

        num_attacks = int(self._rng.integers(*NUM_ATTACKS_PER_STEP_RANGE))

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

            # Attack type distribution depends on kill-chain stage
            if self._stage == STAGE_RECON:
                # Mostly port scans, mild damage
                atk_type = ATTACK_DOS if self._rng.random() < 0.7 else ATTACK_BRUTE_FORCE
                raw = float(self._rng.uniform(0.03, 0.08))
            elif self._stage == STAGE_EXPLOIT:
                # Brute force + CVE exploits, medium damage
                if h.vulnerabilities and self._rng.random() < 0.6:
                    atk_type = ATTACK_EXPLOIT
                else:
                    atk_type = ATTACK_BRUTE_FORCE
                raw = float(self._rng.uniform(0.12, 0.22))
            else:  # STAGE_ESCALATE
                # Strong exploits, high damage
                if h.vulnerabilities and self._rng.random() < 0.7:
                    atk_type = ATTACK_EXPLOIT
                else:
                    atk_type = ATTACK_DOS
                raw = float(self._rng.uniform(0.18, 0.32))

            attacks.append(Attack(
                target_id=h.id,
                attack_type=atk_type,
                stage=self._stage,
                raw_damage=raw,
            ))

        return attacks

    def _resolve_attacks(self) -> Dict[str, Any]:
        events = []
        total_raw = 0.0
        total_applied = 0.0
        total_prevented = 0.0

        # Reset per-host attack flags
        self._under_attack = [False] * self.n_hosts

        for atk in self._pending_attacks:
            host = self.net.hosts[atk.target_id]
            self._under_attack[atk.target_id] = True

            multiplier = 1.0
            reasons = []

            if atk.attack_type == ATTACK_EXPLOIT:
                if not host.vulnerabilities:
                    multiplier *= 0.25
                    reasons.append("no_cves")
                else:
                    multiplier *= 0.6 + 0.4 * min(len(host.vulnerabilities), 2) / 2
                    reasons.append(f"n_cves={len(host.vulnerabilities)}")

            elif atk.attack_type in (ATTACK_DOS, ATTACK_BRUTE_FORCE):
                if not host.open_ports:
                    multiplier *= 0.10
                    reasons.append("no_ports")
                else:
                    multiplier *= 0.5 + 0.5 * min(len(host.open_ports), 3) / 3
                    reasons.append(f"n_ports={len(host.open_ports)}")

            applied = atk.raw_damage * multiplier
            prevented = atk.raw_damage - applied

            if host.role != "router" and not host.open_ports:
                applied *= 0.3
                prevented = atk.raw_damage - applied
                reasons.append("isolated")

            host.take_damage(applied)
            total_raw += atk.raw_damage
            total_applied += applied
            total_prevented += prevented

            events.append({
                "target": atk.target_id,
                "type": int(atk.attack_type),
                "stage": int(atk.stage),
                "raw": round(atk.raw_damage, 3),
                "applied": round(applied, 3),
                "prevented": round(prevented, 3),
                "reasons": reasons,
            })
            self.net.log_event("attack", **events[-1])

        self._last_damage_events = events
        return {
            "n_attacks": len(events),
            "total_raw": round(total_raw, 3),
            "total_applied": round(total_applied, 3),
            "total_prevented": round(total_prevented, 3),
            "events": events,
        }

    def _apply_recovery(self) -> None:
        for h in self.net.hosts.values():
            if h.compromised:
                continue
            if h.health < 1.0:
                h.health = min(1.0, h.health + RECOVERY_PER_STEP)

    # ---------------- reward & termination ----------------

    def _compute_reward(self, action_type, action_effect, attack_effect) -> float:
        hosts = list(self.net.hosts.values())
        n_healthy = sum(1 for h in hosts if not h.compromised)
        n_compromised = sum(1 for h in hosts if h.compromised)

        reward = 0.0
        reward += R_PER_HEALTHY_STEP * n_healthy
        reward += R_PER_COMPROMISED * n_compromised
        reward += R_DAMAGE_PREVENTED * attack_effect["total_prevented"]

        if action_type != ACTION_NOOP and action_effect.get("applied"):
            reward += R_ACTION_COST

        return reward

    def _check_termination(self) -> Tuple[bool, bool]:
        hosts = list(self.net.hosts.values())
        n_healthy = sum(1 for h in hosts if not h.compromised)
        if n_healthy == 0:
            return True, False
        if n_healthy <= LOSS_THRESHOLD:
            return True, False
        return False, False

    # ---------------- observation ----------------

    def _get_obs(self) -> np.ndarray:
        # Host vectors (in id order)
        ids_sorted = sorted(self.net.hosts.keys())
        host_vecs = np.concatenate(
            [self.net.hosts[i].to_vector() for i in ids_sorted]
        )

        # Network summary
        hosts = list(self.net.hosts.values())
        avg_health = float(np.mean([h.health for h in hosts]))
        n_compromised = float(sum(1 for h in hosts if h.compromised))
        n_healthy = float(sum(1 for h in hosts if not h.compromised))
        traffic_sum = float(sum(h.traffic_rate for h in hosts))
        summary = np.array([avg_health, n_compromised, n_healthy, traffic_sum],
                           dtype=np.float32)

        # Reserved
        reserved = np.zeros(RESERVED_SIZE, dtype=np.float32)

        # Last action one-hot
        last_action_vec = np.zeros(N_ACTION_TYPES, dtype=np.float32)
        if 0 <= self._last_action < N_ACTION_TYPES:
            last_action_vec[self._last_action] = 1.0

        # Last target one-hot
        last_target_vec = np.zeros(N_HOSTS, dtype=np.float32)
        if 0 <= self._last_target < N_HOSTS:
            last_target_vec[self._last_target] = 1.0

        # Stage one-hot (recon=0, exploit=1, escalate=2)
        stage_vec = np.zeros(ATTACK_STAGE_SIZE, dtype=np.float32)
        if self._stage == STAGE_RECON:
            stage_vec[0] = 1.0
        elif self._stage == STAGE_EXPLOIT:
            stage_vec[1] = 1.0
        elif self._stage == STAGE_ESCALATE:
            stage_vec[2] = 1.0

        # Per-host attack alert flags
        alert_vec = np.array(self._under_attack, dtype=np.float32)

        obs = np.concatenate([
            host_vecs, summary, reserved,
            last_action_vec, last_target_vec, stage_vec, alert_vec,
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "stage": int(self._stage),
            "n_compromised": len(self.net.compromised_hosts()),
            "n_healthy": len(self.net.healthy_hosts()),
            "last_attacks": self._last_damage_events[-3:],
            "recent_events": self.net.recent_events(3),
        }


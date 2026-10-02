"""
CyberDefenseEnv - Gymnasium environment for the ACDS project (v2).

V2 CHANGES (Day 7):
- Attacks now target vulnerabilities and open ports, so defense actions matter.
- Blocking a port reduces damage from port-based attacks.
- Patching a CVE reduces damage from CVE-based attacks.
- Healthy hosts slowly regenerate health (recovery).
- Reward includes damage-prevented partial credit.
- Win condition: survive max_steps with >= 7 healthy hosts.
- Lose condition: all hosts compromised (terminated) OR <= 2 healthy (terminated).

The goal of V2: create a clean causal chain from action -> reward.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.sim.host import Host
from src.sim.network import Network


# ---------- configuration ----------

N_HOSTS = 10
MAX_STEPS = 100

# Action ids (keep stable)
ACTION_NOOP       = 0
ACTION_PATCH      = 1
ACTION_BLOCK_PORT = 2
ACTION_ISOLATE    = 3
ACTION_SCAN       = 4
N_ACTIONS = 5

# Attack types
ATTACK_DOS         = "dos"           # exploits open ports
ATTACK_BRUTE_FORCE = "brute_force"   # exploits open SSH/RDP ports
ATTACK_EXPLOIT     = "exploit"       # exploits known CVEs
ATTACK_TYPES = [ATTACK_DOS, ATTACK_BRUTE_FORCE, ATTACK_EXPLOIT]

# Reward shaping
R_PER_HEALTHY_STEP    =  +0.05   # per healthy host per step (survival)
R_PER_COMPROMISED     =  -0.20   # per compromised host per step
R_DAMAGE_PREVENTED    =  +0.50   # bonus * damage absorbed by defense
R_ACTION_COST         =  -0.02   # small cost per non-noop action
R_TERMINAL_WIN        = +20.0    # survive with >= 7 healthy hosts
R_TERMINAL_LOSS       = -20.0    # all hosts compromised

# Damage model
BASE_ATTACK_DAMAGE = 0.15        # raw damage before defense
RECOVERY_PER_STEP  = 0.02        # healthy hosts regen this much per step
NUM_ATTACKS_PER_STEP_RANGE = (1, 3)   # 1-2 attacks per step

# Vulnerability pool (assigned to hosts at reset)
CVE_POOL = [
    "CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003",
    "CVE-2024-0004", "CVE-2024-0005", "CVE-2024-0006",
]


@dataclass
class Attack:
    """A pending attack that will be resolved this step."""
    target_id: int
    attack_type: str
    raw_damage: float


class CyberDefenseEnv(gym.Env):
    """Simulated network where an RL agent defends hosts against attacks (v2)."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_hosts: int = N_HOSTS, max_steps: int = MAX_STEPS):
        super().__init__()
        self.n_hosts = n_hosts
        self.max_steps = max_steps
        self._step_count = 0

        self.net = self._build_network(n_hosts)

        obs_len = Network.vector_size(n_hosts)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._rng = np.random.default_rng()

        # Per-step scratch data
        self._pending_attacks: List[Attack] = []
        self._last_damage_events: List[Dict[str, Any]] = []

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
        """Give each non-router host 1-2 random CVEs at episode start."""
        for h in self.net.hosts.values():
            h.vulnerabilities = []
        candidates = [h for h in self.net.hosts.values() if h.role != "router"]
        for h in candidates:
            n_cve = int(self._rng.integers(1, 3))   # 1 or 2
            h.vulnerabilities = list(
                self._rng.choice(CVE_POOL, size=n_cve, replace=False)
            )

    # ---------------- gym API ----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self.net.reset()
        self._step_count = 0
        self._pending_attacks = []
        self._last_damage_events = []
        self._assign_random_vulnerabilities()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        self._step_count += 1
        self.net.current_step = self._step_count

        # 1. Agent acts first (defenses applied)
        action_effect = self._apply_action(int(action))

        # 2. Generate this step's attacks BEFORE resolving them,
        #    so we can compute damage-prevented at resolution time.
        self._pending_attacks = self._generate_attacks()

        # 3. Resolve attacks: defense state filters damage
        attack_effect = self._resolve_attacks()

        # 4. Recovery for healthy, uncompromised hosts
        self._apply_recovery()

        # 5. Reward & termination
        reward = self._compute_reward(action, action_effect, attack_effect)

        terminated, win = self._check_termination()
        if terminated:
            reward += R_TERMINAL_WIN if win else R_TERMINAL_LOSS

        truncated = self._step_count >= self.max_steps
        # Win condition on truncation: >= 7 healthy hosts
        if truncated:
            n_healthy = sum(1 for h in self.net.hosts.values() if not h.compromised)
            if n_healthy >= 7:
                reward += R_TERMINAL_WIN
                terminated = True  # count as a terminal win

        obs = self._get_obs()
        info = self._get_info()
        info["action_effect"] = action_effect
        info["attack_effect"] = attack_effect
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        comp = [h.id for h in self.net.compromised_hosts()]
        avg_h = np.mean([h.health for h in self.net.hosts.values()])
        print(f"[step {self._step_count:3d}] compromised={comp} avg_health={avg_h:.2f}")

    # ---------------- action effects ----------------

    def _apply_action(self, action: int) -> Dict[str, Any]:
        """Apply agent's action. Returns a dict describing what happened."""
        effect: Dict[str, Any] = {"action": action, "applied": False}

        if action == ACTION_NOOP:
            effect["applied"] = True
            return effect

        target = self._most_at_risk_host()
        if target is None:
            return effect

        if action == ACTION_PATCH:
            if target.vulnerabilities:
                cve = self._rng.choice(target.vulnerabilities)
                target.patch(cve)
                effect.update(applied=True, target=target.id, removed=cve)
            else:
                effect.update(applied=False, target=target.id, reason="no_vulns")

        elif action == ACTION_BLOCK_PORT:
            if target.open_ports:
                port = int(self._rng.choice(target.open_ports))
                target.open_ports.remove(port)
                effect.update(applied=True, target=target.id, blocked_port=port)
            else:
                effect.update(applied=False, target=target.id, reason="no_ports")

        elif action == ACTION_ISOLATE:
            removed = list(target.open_ports)
            target.open_ports = []
            target.take_damage(0.03)   # small cost
            effect.update(applied=True, target=target.id, removed_ports=removed)

        elif action == ACTION_SCAN:
            effect.update(
                applied=True,
                target=target.id,
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

    # ---------------- attack generation & resolution ----------------

    def _generate_attacks(self) -> List[Attack]:
        """Generate attacks. Preference: hosts with unpatched CVEs or open ports."""
        hosts = [h for h in self.net.hosts.values()]
        if not hosts:
            return []

        num_attacks = int(self._rng.integers(*NUM_ATTACKS_PER_STEP_RANGE))

        # Weight each host by its "attackability"
        weights = []
        for h in hosts:
            w = 1.0
            w += 1.5 * len(h.vulnerabilities)             # CVEs attract exploits
            w += 0.5 * len(h.open_ports)                  # more ports = more surface
            if h.compromised:
                w *= 0.3                                  # less reason to re-attack
            weights.append(w)
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()

        chosen = self._rng.choice(len(hosts), size=num_attacks,
                                  replace=True, p=weights)

        attacks: List[Attack] = []
        for idx in chosen:
            h = hosts[int(idx)]
            # Attack type chosen based on host's surface
            if h.vulnerabilities and (self._rng.random() < 0.6):
                atk = ATTACK_EXPLOIT
            elif 22 in h.open_ports or 3389 in h.open_ports:
                atk = ATTACK_BRUTE_FORCE
            else:
                atk = ATTACK_DOS
            raw = float(self._rng.uniform(0.10, 0.20))
            attacks.append(Attack(target_id=h.id, attack_type=atk, raw_damage=raw))

        return attacks

    def _resolve_attacks(self) -> Dict[str, Any]:
        """Apply attacks, but let defense state reduce damage. Track damage prevented."""
        events = []
        total_raw = 0.0
        total_applied = 0.0
        total_prevented = 0.0

        for atk in self._pending_attacks:
            host = self.net.hosts[atk.target_id]
            multiplier = 1.0
            reasons = []

            # CVE exploit: reduce damage by fraction of CVEs patched
            if atk.attack_type == ATTACK_EXPLOIT:
                if not host.vulnerabilities:
                    multiplier *= 0.25
                    reasons.append("no_cves")
                else:
                    multiplier *= 0.6 + 0.4 * min(len(host.vulnerabilities), 2) / 2
                    reasons.append(f"n_cves={len(host.vulnerabilities)}")

            # Port-based: reduce if ports closed
            elif atk.attack_type in (ATTACK_DOS, ATTACK_BRUTE_FORCE):
                if not host.open_ports:
                    multiplier *= 0.10
                    reasons.append("no_ports")
                else:
                    multiplier *= 0.5 + 0.5 * min(len(host.open_ports), 3) / 3
                    reasons.append(f"n_ports={len(host.open_ports)}")

            applied = atk.raw_damage * multiplier
            prevented = atk.raw_damage - applied

            # Isolation blocks nearly everything
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
                "type": atk.attack_type,
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
            if not h.compromised and h.health < 1.0:
                h.health = min(1.0, h.health + RECOVERY_PER_STEP)

    # ---------------- reward & termination ----------------

    def _compute_reward(self, action, action_effect, attack_effect) -> float:
        hosts = list(self.net.hosts.values())
        n_healthy = sum(1 for h in hosts if not h.compromised)
        n_compromised = sum(1 for h in hosts if h.compromised)

        reward = 0.0
        reward += R_PER_HEALTHY_STEP * n_healthy
        reward += R_PER_COMPROMISED * n_compromised
        reward += R_DAMAGE_PREVENTED * attack_effect["total_prevented"]

        if action != ACTION_NOOP and action_effect.get("applied"):
            reward += R_ACTION_COST

        return reward

    def _check_termination(self) -> Tuple[bool, bool]:
        hosts = list(self.net.hosts.values())
        n_healthy = sum(1 for h in hosts if not h.compromised)
        if n_healthy == 0:
            return True, False   # total loss
        if n_healthy <= 2:
            return True, False   # functional collapse = loss
        return False, False

    # ---------------- observation & info ----------------

    def _get_obs(self) -> np.ndarray:
        v = self.net.to_vector()
        return np.clip(v, 0.0, 1.0).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "n_compromised": len(self.net.compromised_hosts()),
            "n_healthy": len(self.net.healthy_hosts()),
            "last_attacks": self._last_damage_events[-3:],
            "recent_events": self.net.recent_events(3),
        }

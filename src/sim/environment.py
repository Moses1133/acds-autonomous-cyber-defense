"""
CyberDefenseEnv - a Gymnasium environment for the ACDS project.

Observation: fixed-length float32 vector describing the network state.
Action:      Discrete(5) defensive action applied globally.
Reward:      dense - healthy hosts +, compromised hosts -, small action cost.
Episode:     100 steps (truncated) OR all hosts compromised (terminated).

Week 5 note: attacks are random for now. Week 4 of the project replaces
this with real attack classes (DoS, port scan, brute force).
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.sim.host import Host
from src.sim.network import Network


# ---------- configuration ----------

N_HOSTS = 10
MAX_STEPS = 100

# Action ids (keep stable! the RL agent depends on this order)
ACTION_NOOP          = 0
ACTION_PATCH         = 1
ACTION_BLOCK_PORT    = 2
ACTION_ISOLATE       = 3
ACTION_SCAN          = 4
N_ACTIONS = 5

# Reward weights
R_HEALTHY         =  +0.1    # per healthy host per step
R_COMPROMISED     =  -0.5    # per compromised host per step
R_ACTION_COST     =  -0.01   # small cost per non-noop action
R_TERMINAL_LOSS   = -10.0    # extra penalty if all hosts compromised


class CyberDefenseEnv(gym.Env):
    """A tiny simulated network where an RL agent learns to defend hosts."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_hosts: int = N_HOSTS, max_steps: int = MAX_STEPS):
        super().__init__()

        self.n_hosts = n_hosts
        self.max_steps = max_steps
        self._step_count = 0

        # Build the network once; reset() will reinitialize it
        self.net = self._build_network(n_hosts)

        # ---------- spaces ----------
        obs_len = Network.vector_size(n_hosts)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # RNG (seeded in reset)
        self._rng = np.random.default_rng()

    # ---------------- construction ----------------

    def _build_network(self, n: int) -> Network:
        """Create n hosts in a star topology around host 0 (the router)."""
        net = Network()
        net.add_host(Host(id=0, ip="10.0.0.1", role="router",
                          open_ports=[22, 53, 80]))
        for i in range(1, n):
            role = "server" if i % 3 == 0 else "workstation"
            ports = [22, 80, 443] if role == "server" else [22, 3389]
            net.add_host(Host(id=i, ip=f"10.0.0.{i+1}", role=role, open_ports=ports))
            net.add_link(0, i)   # star: everything connects to the router
        return net

    # ---------------- gym API ----------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self.net.reset()
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        self.net.current_step = self._step_count

        # 1. Apply the agent's action
        action_effect = self._apply_action(int(action))

        # 2. Random attack happens (placeholder for Week 4's real attacks)
        attack_effect = self._inject_random_attack()

        # 3. Compute reward from new state
        reward = self._compute_reward(action, action_effect)

        # 4. Check termination
        terminated = self._all_hosts_compromised()
        if terminated:
            reward += R_TERMINAL_LOSS
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["action_effect"] = action_effect
        info["attack_effect"] = attack_effect
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        comp = [h.id for h in self.net.compromised_hosts()]
        print(f"[step {self._step_count:3d}] compromised={comp} "
              f"avg_health={np.mean([h.health for h in self.net.hosts.values()]):.2f}")

    # ---------------- action effects ----------------

    def _apply_action(self, action: int) -> Dict[str, Any]:
        """Apply the agent's action. Return a dict describing what happened."""
        effect: Dict[str, Any] = {"action": action, "applied": False}

        if action == ACTION_NOOP:
            effect["applied"] = True
            return effect

        # For now, actions apply to the "most at-risk" non-router host
        target = self._most_at_risk_host()
        if target is None:
            return effect  # nothing to defend

        if action == ACTION_PATCH:
            # Remove a random vulnerability if present
            if target.vulnerabilities:
                cve = self._rng.choice(target.vulnerabilities)
                target.patch(cve)
                effect.update(applied=True, target=target.id, removed=cve)
            else:
                effect.update(applied=False, target=target.id, reason="no_vulns")

        elif action == ACTION_BLOCK_PORT:
            if target.open_ports:
                port = self._rng.choice(target.open_ports)
                target.open_ports.remove(int(port))
                effect.update(applied=True, target=target.id, blocked_port=int(port))
            else:
                effect.update(applied=False, target=target.id, reason="no_ports")

        elif action == ACTION_ISOLATE:
            # Isolation = drop all ports + small health cost (side effect)
            removed = list(target.open_ports)
            target.open_ports = []
            target.take_damage(0.05)
            effect.update(applied=True, target=target.id, removed_ports=removed)

        elif action == ACTION_SCAN:
            # Scan reveals hidden state; for now just returns info
            effect.update(
                applied=True,
                target=target.id,
                revealed={
                    "health": target.health,
                    "compromised": target.compromised,
                    "n_vulns": len(target.vulnerabilities),
                },
            )

        self.net.log_event("defense", **effect)
        return effect

    def _most_at_risk_host(self) -> Optional[Host]:
        """Return the non-router host with the lowest health (tie: lowest id)."""
        candidates = [h for h in self.net.hosts.values() if h.role != "router"]
        if not candidates:
            return None
        return min(candidates, key=lambda h: (h.health, h.id))

    # ---------------- attack generator (placeholder) ----------------

    def _inject_random_attack(self) -> Dict[str, Any]:
        """
        Pick 1-2 random hosts and damage them. Placeholder until Week 4
        when we add structured attacks (DoS, port scan, brute force).
        """
        num_targets = self._rng.integers(1, 3)  # 1 or 2
        ids = list(self.net.hosts.keys())
        targets = self._rng.choice(ids, size=num_targets, replace=False)

        effects = []
        for tid in targets:
            host = self.net.hosts[int(tid)]
            damage = float(self._rng.uniform(0.05, 0.25))
            host.take_damage(damage)
            effects.append({"target": int(tid), "damage": round(damage, 3)})
            self.net.log_event("attack", target=int(tid), damage=damage)

        return {"targets": effects}

    # ---------------- reward & termination ----------------

    def _compute_reward(self, action: int, action_effect: Dict[str, Any]) -> float:
        hosts = list(self.net.hosts.values())
        n_healthy = sum(1 for h in hosts if not h.compromised)
        n_compromised = sum(1 for h in hosts if h.compromised)

        reward = R_HEALTHY * n_healthy + R_COMPROMISED * n_compromised

        if action != ACTION_NOOP and action_effect.get("applied"):
            reward += R_ACTION_COST

        return reward

    def _all_hosts_compromised(self) -> bool:
        return all(h.compromised for h in self.net.hosts.values())

    # ---------------- observation & info ----------------

    def _get_obs(self) -> np.ndarray:
        v = self.net.to_vector()
        # Everything is already in [0, 1] by construction, but clip for safety
        return np.clip(v, 0.0, 1.0).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "n_compromised": len(self.net.compromised_hosts()),
            "n_healthy": len(self.net.healthy_hosts()),
            "recent_events": self.net.recent_events(3),
        }

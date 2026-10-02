"""
Host - the atomic unit of the simulated network.

A Host represents one device (server, workstation, router) with:
- identity:     id, ip, role
- state:        health, open_ports, traffic_rate, compromised, vulnerabilities
- behavior:     take_damage, patch, reset, to_vector

Design rules:
1. State in attributes, behavior in methods. No cross-object mutations.
2. Every stateful object can flatten itself into a numeric vector via to_vector().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


# Fixed vocabulary for roles - keeps vectors consistent across hosts
ROLES = ["server", "workstation", "router"]


@dataclass
class Host:
    """A single device in the simulated network."""

    id: int
    ip: str
    role: str

    # --- Numeric state ---
    health: float = 1.0                  # 0.0 = dead, 1.0 = fully healthy
    traffic_rate: float = 0.0            # packets per second (arbitrary units)

    # --- Discrete state ---
    open_ports: List[int] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    compromised: bool = False

    # ---------------- behavior ----------------

    def take_damage(self, amount: float) -> None:
        """Reduce health. Clamp to [0, 1]. Mark as compromised if health dips low."""
        self.health = max(0.0, self.health - amount)
        if self.health < 0.3:
            self.compromised = True

    def patch(self, cve: str) -> bool:
        """Remove a vulnerability. Return True if it was present."""
        if cve in self.vulnerabilities:
            self.vulnerabilities.remove(cve)
            return True
        return False

    def reset(self) -> None:
        """Restore to initial healthy state (used by env.reset())."""
        self.health = 1.0
        self.traffic_rate = 0.0
        self.compromised = False
        # open_ports and vulnerabilities persist - they are config, not runtime state

    # ---------------- introspection ----------------

    def to_vector(self) -> np.ndarray:
        """
        Flatten host state into a fixed-length numeric vector.
        Order MUST stay stable across versions - the RL agent depends on it.

        Layout (length 4 + 3 = 7):
          [health, traffic_rate, compromised, num_open_ports,
           role_onehot(server), role_onehot(workstation), role_onehot(router)]
        """
        role_onehot = [1.0 if self.role == r else 0.0 for r in ROLES]
        return np.array(
            [
                self.health,
                self.traffic_rate,
                1.0 if self.compromised else 0.0,
                float(len(self.open_ports)),
                *role_onehot,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def vector_size() -> int:
        """Length of the vector returned by to_vector(). Useful for building obs spaces."""
        return 4 + len(ROLES)

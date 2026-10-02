"""
Network - a collection of hosts with topology, traffic, and an event log.

Responsibilities:
- Own a set of Host objects (indexed by id)
- Track links (undirected edges) between hosts
- Maintain a rolling event log (attacks, defenses, state changes)
- Provide a fixed-length numeric vector summarizing the whole network

Design rule (Day 3): the vector length is a stable contract.
If we ever change it, bump VECTOR_VERSION and retrain.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from src.sim.host import Host


VECTOR_VERSION = 1

# How many floats the network summary adds on top of the host vectors
NETWORK_SUMMARY_SIZE = 4
# Reserved for future expansion - always output zeros for now
RESERVED_SIZE = 2


@dataclass
class Network:
    """A set of hosts + topology + shared state."""

    hosts: Dict[int, Host] = field(default_factory=dict)

    # Undirected edges. Stored canonically as (min_id, max_id).
    links: List[Tuple[int, int]] = field(default_factory=list)

    # Rolling event log: list of (t, kind, payload) tuples.
    # 't' is the simulation step when the event happened.
    events: List[Tuple[int, str, dict]] = field(default_factory=list)

    # Global step counter (incremented by the env, not by the network itself)
    current_step: int = 0

    # ---------------- construction ----------------

    def add_host(self, host: Host) -> None:
        if host.id in self.hosts:
            raise ValueError(f"Host id {host.id} already exists")
        self.hosts[host.id] = host

    def add_link(self, a_id: int, b_id: int) -> None:
        """Add an undirected link. Idempotent."""
        if a_id == b_id:
            raise ValueError("Self-links are not allowed")
        if a_id not in self.hosts or b_id not in self.hosts:
            raise KeyError("Both hosts must exist before linking")
        edge = (min(a_id, b_id), max(a_id, b_id))
        if edge not in self.links:
            self.links.append(edge)

    # ---------------- queries ----------------

    def neighbors(self, host_id: int) -> List[int]:
        out = []
        for a, b in self.links:
            if a == host_id:
                out.append(b)
            elif b == host_id:
                out.append(a)
        return out

    def n_hosts(self) -> int:
        return len(self.hosts)

    def healthy_hosts(self) -> List[Host]:
        return [h for h in self.hosts.values() if not h.compromised]

    def compromised_hosts(self) -> List[Host]:
        return [h for h in self.hosts.values() if h.compromised]

    # ---------------- event log ----------------

    def log_event(self, kind: str, **payload) -> None:
        """Append an event tagged with the current step."""
        self.events.append((self.current_step, kind, payload))

    def recent_events(self, n: int = 5) -> List[Tuple[int, str, dict]]:
        return self.events[-n:]

    # ---------------- vectorization ----------------

    def to_vector(self) -> np.ndarray:
        """
        Fixed-length float vector describing the whole network.

        Layout:
          [host_0_vector, host_1_vector, ..., host_{N-1}_vector,   (N * host_vec_size)
           avg_health, n_compromised, n_healthy, traffic_sum,      (NETWORK_SUMMARY_SIZE)
           reserved_0, reserved_1]                                 (RESERVED_SIZE)
        """
        if not self.hosts:
            raise RuntimeError("Network has no hosts")

        # Sort by id so order is deterministic
        ids_sorted = sorted(self.hosts.keys())

        host_vecs = np.concatenate(
            [self.hosts[i].to_vector() for i in ids_sorted]
        )

        hosts = list(self.hosts.values())
        avg_health = float(np.mean([h.health for h in hosts]))
        n_compromised = float(sum(1 for h in hosts if h.compromised))
        n_healthy = float(sum(1 for h in hosts if not h.compromised))
        traffic_sum = float(sum(h.traffic_rate for h in hosts))

        summary = np.array(
            [avg_health, n_compromised, n_healthy, traffic_sum],
            dtype=np.float32,
        )

        reserved = np.zeros(RESERVED_SIZE, dtype=np.float32)

        return np.concatenate([host_vecs, summary, reserved]).astype(np.float32)

    @staticmethod
    def vector_size(n_hosts: int) -> int:
        """Compute the expected length of to_vector() for a network of n_hosts."""
        return n_hosts * Host.vector_size() + NETWORK_SUMMARY_SIZE + RESERVED_SIZE

    # ---------------- helpers ----------------

    def reset(self) -> None:
        """Reset all hosts and clear the event log. Keeps topology intact."""
        for h in self.hosts.values():
            h.reset()
        self.events.clear()
        self.current_step = 0

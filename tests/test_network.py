"""
Sanity tests for Network.
Run: python tests\test_network.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sim.host import Host
from src.sim.network import Network, VECTOR_VERSION


def build_small_net() -> Network:
    net = Network()
    net.add_host(Host(id=0, ip="10.0.0.1", role="server",      open_ports=[80, 443]))
    net.add_host(Host(id=1, ip="10.0.0.2", role="workstation", open_ports=[22]))
    net.add_host(Host(id=2, ip="10.0.0.3", role="router",      open_ports=[22, 53]))
    net.add_host(Host(id=3, ip="10.0.0.4", role="workstation", open_ports=[22, 3389]))

    # Star topology around the router (id=2)
    net.add_link(0, 2)
    net.add_link(1, 2)
    net.add_link(3, 2)
    return net


def main():
    print(f"=== Building network (VECTOR_VERSION={VECTOR_VERSION}) ===")
    net = build_small_net()
    print(f"  hosts = {net.n_hosts()}")
    print(f"  links = {net.links}")

    print("\n=== Neighbors ===")
    for h_id in sorted(net.hosts.keys()):
        print(f"  host {h_id}: neighbors={net.neighbors(h_id)}")

    print("\n=== Vector shape contract ===")
    v = net.to_vector()
    expected = Network.vector_size(net.n_hosts())
    print(f"  vector length = {len(v)} (expected {expected})")
    assert len(v) == expected, "vector length mismatch"
    assert v.dtype == np.float32

    print(f"  vector (first 12 floats): {v[:12]}")
    print(f"  vector (last 6 floats):   {v[-6:]}")

    print("\n=== Behavior: damage a host ===")
    net.current_step = 1
    net.hosts[0].take_damage(0.8)     # server goes compromised
    net.log_event("attack", target=0, kind="dos", severity=0.8)
    print(f"  host 0 health = {net.hosts[0].health:.2f}, compromised={net.hosts[0].compromised}")
    print(f"  compromised hosts = {[h.id for h in net.compromised_hosts()]}")
    print(f"  healthy hosts = {[h.id for h in net.healthy_hosts()]}")

    print("\n=== Summary fields moved ===")
    v2 = net.to_vector()
    host_vec_size = net.hosts[0].vector_size()
    n_hosts = net.n_hosts()
    summary = v2[n_hosts * host_vec_size : n_hosts * host_vec_size + 4]
    print(f"  [avg_health, n_compromised, n_healthy, traffic_sum] = {summary}")
    assert summary[1] == 1.0, "one host should be compromised"
    assert summary[2] == 3.0, "three hosts should be healthy"

    print("\n=== Event log ===")
    for e in net.recent_events(3):
        print(f"  {e}")
    assert len(net.events) == 1

    print("\n=== Reset ===")
    net.reset()
    print(f"  host 0 health after reset = {net.hosts[0].health:.2f}")
    print(f"  events after reset = {len(net.events)}")
    assert net.hosts[0].health == 1.0
    assert len(net.events) == 0

    print("\nAll network tests passed.")


if __name__ == "__main__":
    main()

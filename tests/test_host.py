"""
Sanity tests for Host.
Run: python tests\test_host.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sim.host import Host, ROLES


def main():
    print("=== Creating 3 hosts ===")
    h1 = Host(id=0, ip="10.0.0.1", role="server",   open_ports=[80, 443])
    h2 = Host(id=1, ip="10.0.0.2", role="workstation", open_ports=[22])
    h3 = Host(id=2, ip="10.0.0.3", role="router",   open_ports=[22, 53, 80])

    hosts = [h1, h2, h3]
    for h in hosts:
        print(f"  {h}")

    print("\n=== Vectors ===")
    for h in hosts:
        v = h.to_vector()
        assert v.shape == (Host.vector_size(),), f"bad shape: {v.shape}"
        print(f"  host {h.id} ({h.role:11s}) -> {v}")

    print("\n=== Behavior: take_damage ===")
    h1.take_damage(0.25)
    print(f"  after 0.25 dmg -> health={h1.health:.2f}, compromised={h1.compromised}")
    h1.take_damage(0.60)   # 1.00 - 0.25 - 0.60 = 0.15 -> compromised
    print(f"  after 0.60 dmg -> health={h1.health:.2f}, compromised={h1.compromised}")
    assert h1.compromised is True, "host should be compromised below 0.3 health"

    print("\n=== Behavior: patch ===")
    h2.vulnerabilities = ["CVE-2024-0001", "CVE-2024-0002"]
    ok = h2.patch("CVE-2024-0001")
    print(f"  patched CVE-2024-0001? {ok}  remaining={h2.vulnerabilities}")
    ok = h2.patch("CVE-9999-9999")
    print(f"  patched CVE-9999-9999? {ok}  remaining={h2.vulnerabilities}")
    assert ok is False, "patching absent CVE should return False"

    print("\n=== Behavior: reset ===")
    h1.reset()
    print(f"  host 1 after reset -> health={h1.health:.2f}, compromised={h1.compromised}")
    assert h1.health == 1.0 and h1.compromised is False

    print("\n=== Vector sanity ===")
    v = h1.to_vector()
    print(f"  host 1 vector: {v}")
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32
    print(f"  health component = {v[0]:.2f}")
    print(f"  role one-hot = {v[4:]}")

    print("\nAll host tests passed.")


if __name__ == "__main__":
    main()

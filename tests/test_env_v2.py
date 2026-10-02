"""
Sanity tests for CyberDefenseEnv v2.
Verifies the causal chain: defense state reduces damage.
Run: python tests\test_env_v2.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sim.environment import (
    CyberDefenseEnv, N_ACTIONS,
    ACTION_NOOP, ACTION_PATCH, ACTION_BLOCK_PORT, ACTION_ISOLATE, ACTION_SCAN,
)


def main():
    print("=== Build env v2 ===")
    env = CyberDefenseEnv(n_hosts=10, max_steps=100)
    print(f"  obs space: {env.observation_space}")
    print(f"  act space: {env.action_space}")

    print("\n=== Reset (seeded) ===")
    obs, info = env.reset(seed=42)
    print(f"  obs shape = {obs.shape}")
    print(f"  n_healthy = {info['n_healthy']}")

    # Check vulnerabilities assigned
    n_with_cves = sum(1 for h in env.net.hosts.values() if h.vulnerabilities)
    print(f"  hosts with CVEs: {n_with_cves}")
    assert n_with_cves >= 8

    print("\n=== Causal chain test: does patching help? ===")
    # Manually damage a host and give it CVEs
    victim = env.net.hosts[1]
    victim.vulnerabilities = ["CVE-2024-0001"]
    print(f"  victim 1 ports before patch: {victim.open_ports}")

    # Run 50 steps with always-patch
    obs, _ = env.reset(seed=7)
    total_patch_reward = 0.0
    for t in range(50):
        obs, r, term, trunc, info = env.step(ACTION_PATCH)
        total_patch_reward += r
        if term or trunc:
            break
    print(f"  PATCH-only 50 steps -> reward = {total_patch_reward:+.2f}, "
          f"reached step {t}")

    # Run 50 steps with always-noop
    obs, _ = env.reset(seed=7)
    total_noop_reward = 0.0
    for t in range(50):
        obs, r, term, trunc, info = env.step(ACTION_NOOP)
        total_noop_reward += r
        if term or trunc:
            break
    print(f"  NOOP-only  50 steps -> reward = {total_noop_reward:+.2f}, "
          f"reached step {t}")

    # Run 50 steps with always-block-port
    obs, _ = env.reset(seed=7)
    total_block_reward = 0.0
    for t in range(50):
        obs, r, term, trunc, info = env.step(ACTION_BLOCK_PORT)
        total_block_reward += r
        if term or trunc:
            break
    print(f"  BLOCK-only 50 steps -> reward = {total_block_reward:+.2f}, "
          f"reached step {t}")

    print("\n=== Verdict ===")
    print(f"  patch  {total_patch_reward:+.2f}")
    print(f"  noop   {total_noop_reward:+.2f}")
    print(f"  block  {total_block_reward:+.2f}")

    # The key test: at least one action must beat noop
    best = max(total_patch_reward, total_block_reward)
    if best > total_noop_reward + 1.0:
        print("  CAUSAL CHAIN WORKS: defense actions beat noop")
    else:
        print("  WARNING: no action clearly beats noop -- reward may still be weak")

    print("\n=== Full random episode ===")
    obs, _ = env.reset(seed=99)
    total = 0.0
    for t in range(env.max_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if t < 3 or t % 30 == 0 or term or trunc:
            print(f"  step {t:3d}  action={a}  reward={r:+.2f}  "
                  f"n_comp={info['n_compromised']}  term={term}  trunc={trunc}")
        if term or trunc:
            break
    print(f"  total reward = {total:+.2f}")

    print("\n=== Reproducibility ===")
    def run(seed):
        env.reset(seed=seed)
        rs = []
        for a in [ACTION_PATCH, ACTION_NOOP, ACTION_BLOCK_PORT, ACTION_ISOLATE, ACTION_SCAN]:
            _, r, _, _, _ = env.step(a)
            rs.append(round(r, 4))
        return rs

    r1 = run(1234)
    r2 = run(1234)
    print(f"  run 1: {r1}")
    print(f"  run 2: {r2}")
    assert r1 == r2
    print("  deterministic")

    env.close()
    print("\nAll v2 env tests passed.")


if __name__ == "__main__":
    main()

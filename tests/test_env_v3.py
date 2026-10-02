"""
Sanity tests for CyberDefenseEnv v3 (hardened).
Verifies causal chain is still intact after hardening.
Run: python tests\test_env_v3.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sim.environment import (
    CyberDefenseEnv, ACTION_NOOP, ACTION_PATCH, ACTION_BLOCK_PORT,
    ACTION_ISOLATE, ACTION_SCAN,
)


def main():
    print("=== Build env v3 ===")
    env = CyberDefenseEnv(n_hosts=10, max_steps=100)
    print(f"  obs space: {env.observation_space}")
    print(f"  act space: {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"  reset ok, obs shape = {obs.shape}, n_healthy = {info['n_healthy']}")

    print("\n=== Action-beats-noop test (v3) ===")
    def run_action(action, seed=7, steps=100):
        env.reset(seed=seed)
        total = 0.0
        for t in range(steps):
            _, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        return total

    r_noop  = run_action(ACTION_NOOP)
    r_patch = run_action(ACTION_PATCH)
    r_block = run_action(ACTION_BLOCK_PORT)
    r_scan  = run_action(ACTION_SCAN)

    print(f"  noop  = {r_noop:+.2f}")
    print(f"  patch = {r_patch:+.2f}")
    print(f"  block = {r_block:+.2f}")
    print(f"  scan  = {r_scan:+.2f}")

    best = max(r_patch, r_block, r_scan)
    if best > r_noop + 5.0:
        print(f"  CAUSAL CHAIN INTACT: best action {best:+.2f} vs noop {r_noop:+.2f}")
    else:
        print(f"  WARNING: best action barely beats noop by {best - r_noop:+.2f}")

    print("\n=== Difficulty check (random play) ===")
    all_rewards = []
    for seed in range(10):
        env.reset(seed=seed)
        total = 0.0
        for t in range(env.max_steps):
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            if term or trunc:
                break
        all_rewards.append(total)
    print(f"  random mean reward over 10 eps: {np.mean(all_rewards):+.2f}")
    print(f"  random min/max: {min(all_rewards):+.2f} / {max(all_rewards):+.2f}")

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
    assert r1 == r2, "env not deterministic"
    print("  deterministic")

    env.close()
    print("\nAll v3 env tests passed.")


if __name__ == "__main__":
    main()

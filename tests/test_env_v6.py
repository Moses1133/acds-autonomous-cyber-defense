"""
Sanity tests for CyberDefenseEnvV6 (MultiDiscrete actions + kill chains).
Run: python tests\test_env_v6.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sim.environment_v6 import (
    CyberDefenseEnvV6, obs_size,
    ACTION_NOOP, ACTION_PATCH, ACTION_BLOCK_PORT, ACTION_ISOLATE, ACTION_SCAN,
    STAGE_RECON, STAGE_EXPLOIT, STAGE_ESCALATE,
)


def main():
    print("=== Build env v6 ===")
    env = CyberDefenseEnvV6(n_hosts=10, max_steps=100)
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")
    print(f"  obs_size helper:   {obs_size(10)}")
    assert env.observation_space.shape == (obs_size(10),)
    assert list(env.action_space.nvec) == [5, 10]

    print("\n=== Reset (seeded) ===")
    obs, info = env.reset(seed=42)
    print(f"  obs shape = {obs.shape}, dtype = {obs.dtype}")
    print(f"  n_healthy = {info['n_healthy']}, stage = {info['stage']}")
    assert obs.shape == env.observation_space.shape

    print("\n=== MultiDiscrete action test ===")
    for action_type, name in [(ACTION_NOOP, "noop"),
                              (ACTION_PATCH, "patch"),
                              (ACTION_BLOCK_PORT, "block_port"),
                              (ACTION_ISOLATE, "isolate"),
                              (ACTION_SCAN, "scan")]:
        env.reset(seed=7)
        obs, r, term, trunc, info = env.step([action_type, 3])
        eff = info["action_effect"]
        print(f"  action={name:11s} target=3 applied={eff.get('applied')} "
              f"reward={r:+.2f}")
        assert eff["target"] == 3
        assert eff["action_type"] == action_type

    print("\n=== Kill chain stages ===")
    env.reset(seed=7)
    stages_seen = set()
    for t in range(100):
        # Fixed action to keep test deterministic
        a = [ACTION_BLOCK_PORT, 3]
        _, r, term, trunc, info = env.step(a)
        stages_seen.add(info["stage"])
        if t in [5, 35, 75]:
            print(f"  step {t:3d}: stage={info['stage']}")
        if term or trunc:
            break
    print(f"  stages seen: {sorted(stages_seen)}")
    assert STAGE_RECON in stages_seen
    assert STAGE_EXPLOIT in stages_seen

    print("\n=== Observation dimensions ===")
    obs, _ = env.reset(seed=1)
    print(f"  expected obs size: 104, actual: {len(obs)}")
    assert len(obs) == 104
    assert 0.0 <= obs.min() and obs.max() <= 1.0

    print("\n=== Reproducibility (fixed actions) ===")
    FIXED_ACTIONS = [
        [ACTION_BLOCK_PORT, 3],
        [ACTION_PATCH, 5],
        [ACTION_NOOP, 0],
        [ACTION_SCAN, 7],
        [ACTION_ISOLATE, 2],
    ] * 2  # 10 actions total

    def run(seed):
        env.reset(seed=seed)
        rewards = []
        for a in FIXED_ACTIONS:
            _, r, term, trunc, _ = env.step(a)
            rewards.append(round(r, 4))
            if term or trunc:
                break
        return rewards

    r1 = run(1234)
    r2 = run(1234)
    print(f"  run 1: {r1}")
    print(f"  run 2: {r2}")
    assert r1 == r2, "env is not deterministic given a seed and fixed actions"
    print("  deterministic")

    print("\n=== Action-beats-noop (rough) ===")
    def run_policy(policy_fn, seed=7, steps=100):
        env.reset(seed=seed)
        total = 0.0
        for t in range(steps):
            a = policy_fn()
            _, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        return total

    r_noop = run_policy(lambda: [ACTION_NOOP, 0])
    r_block = run_policy(lambda: [ACTION_BLOCK_PORT, 5])
    r_scan = run_policy(lambda: [ACTION_SCAN, 5])
    print(f"  noop  = {r_noop:+.2f}")
    print(f"  block = {r_block:+.2f}")
    print(f"  scan  = {r_scan:+.2f}")
    if max(r_block, r_scan) > r_noop + 5:
        print("  CAUSAL CHAIN INTACT: some actions beat noop")
    else:
        print("  WARNING: actions don't beat noop by much")

    env.close()
    print("\nAll v6 env tests passed.")


if __name__ == "__main__":
    main()

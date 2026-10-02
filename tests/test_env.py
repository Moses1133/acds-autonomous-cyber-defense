"""
Sanity tests for CyberDefenseEnv.
Runs a full random-action episode and prints a compact trace.
Run: python tests\test_env.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sim.environment import CyberDefenseEnv, N_ACTIONS


def main():
    print("=== Building env (10 hosts, 100 steps max) ===")
    env = CyberDefenseEnv(n_hosts=10, max_steps=100)
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")

    print("\n=== Reset (seeded) ===")
    obs, info = env.reset(seed=42)
    print(f"  obs shape = {obs.shape}, dtype = {obs.dtype}")
    print(f"  obs min/max = {obs.min():.2f} / {obs.max():.2f}")
    print(f"  info = {info}")
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

    print("\n=== Random episode ===")
    total_reward = 0.0
    for t in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if t < 5 or t % 20 == 0 or terminated or truncated:
            print(f"  step {t:3d}  action={action}  reward={reward:+.2f}  "
                  f"n_comp={info['n_compromised']:2d}  "
                  f"term={terminated}  trunc={truncated}")

        assert obs.shape == env.observation_space.shape
        assert 0.0 <= obs.min() and obs.max() <= 1.0

        if terminated or truncated:
            break

    print(f"\n  Episode ended at step {t}")
    print(f"  Total reward: {total_reward:+.2f}")

    print("\n=== Reproducibility check ===")
    obs1, _ = env.reset(seed=123)
    a1 = [0, 1, 2, 3, 4]
    r1 = []
    for a in a1:
        _, r, _, _, _ = env.step(a)
        r1.append(r)

    obs2, _ = env.reset(seed=123)
    r2 = []
    for a in a1:
        _, r, _, _, _ = env.step(a)
        r2.append(r)

    print(f"  run 1 rewards: {[round(x, 4) for x in r1]}")
    print(f"  run 2 rewards: {[round(x, 4) for x in r2]}")
    assert r1 == r2, "env is not deterministic given a seed"
    print("  seeds match - env is deterministic")

    env.close()
    print("\nAll env tests passed.")


if __name__ == "__main__":
    main()

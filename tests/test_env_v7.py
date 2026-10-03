"""Quick test: v7 with CICIDS-calibrated attacks."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.sim.environment_v7 import CyberDefenseEnvV7

print("=== Building v7 env ===")
env = CyberDefenseEnvV7(n_hosts=10, max_steps=100)
print(f"  obs space: {env.observation_space}")
print(f"  act space: {env.action_space}")
print()
print(env.cicids.summary())
print()

print("=== Running one episode with random actions ===")
obs, info = env.reset(seed=42)
total = 0.0
for t in range(100):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    total += r
    if t in [0, 30, 60, 99]:
        print(f"  step {t:3d}: stage={info['stage']} "
              f"n_compromised={info['n_compromised']} reward={r:+.2f}")
    if term or trunc:
        break

print(f"\nTotal episode reward: {total:+.2f}")
print(f"Final: {info['n_compromised']} compromised, {info['n_healthy']} healthy")
print("\nv7 works.")

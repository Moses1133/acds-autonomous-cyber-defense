"""
Evaluate a trained PPO model on CyberDefenseEnv.

Run: python scripts\evaluate_ppo.py

Loads models\ppo_acds_v1.zip (best) or models\ppo_acds_v1_final.zip.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO

from src.sim.environment import CyberDefenseEnv


N_EPISODES = 5


def run_episode(model, env, render=False):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    for t in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps = t + 1
        if render:
            env.render()
        if terminated or truncated:
            break
    return total_reward, steps, info


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(root, "models", "ppo_acds_v1.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(root, "models", "ppo_acds_v1_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found. Run `python scripts\\train_ppo.py` first."
        )

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    env = CyberDefenseEnv(n_hosts=10, max_steps=100)

    print(f"\n=== Running {N_EPISODES} evaluation episodes ===")
    rewards, lengths, compromised = [], [], []
    for ep in range(N_EPISODES):
        r, length, info = run_episode(model, env, render=False)
        rewards.append(r); lengths.append(length)
        compromised.append(info["n_compromised"])
        print(f"  episode {ep+1}: reward={r:+.2f}  length={length:3d}  "
              f"n_compromised={info['n_compromised']}")

    print("\n=== Summary ===")
    print(f"  mean reward: {np.mean(rewards):+.2f}  (std {np.std(rewards):.2f})")
    print(f"  mean length: {np.mean(lengths):.1f}")
    print(f"  mean compromised at end: {np.mean(compromised):.2f}")

    print("\n=== Random baseline for comparison ===")
    rand_rewards = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=1000 + ep)
        total = 0.0
        for t in range(env.max_steps):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        rand_rewards.append(total)
    print(f"  random mean reward: {np.mean(rand_rewards):+.2f}")

    env.close()


if __name__ == "__main__":
    main()

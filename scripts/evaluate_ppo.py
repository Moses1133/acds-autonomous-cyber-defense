"""
Evaluate a trained PPO model on CyberDefenseEnv.

Also prints action distribution so we can detect policy collapse.

Run: python scripts\evaluate_ppo.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from src.sim.environment import CyberDefenseEnv


N_EPISODES = 20


def run_episode(model, env, seed, record_actions=False):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    actions = []
    for t in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        actions.append(a)
        obs, reward, terminated, truncated, info = env.step(a)
        total_reward += reward
        steps = t + 1
        if terminated or truncated:
            break
    return total_reward, steps, info, actions


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

    print(f"\n=== Trained agent: {N_EPISODES} episodes ===")
    rewards, lengths, compromised = [], [], []
    all_actions = []
    for ep in range(N_EPISODES):
        r, length, info, acts = run_episode(model, env, seed=1000 + ep,
                                             record_actions=True)
        rewards.append(r); lengths.append(length)
        compromised.append(info["n_compromised"])
        all_actions.extend(acts)
        if ep < 5:
            print(f"  episode {ep+1:2d}: reward={r:+7.2f}  length={length:3d}  "
                  f"n_compromised={info['n_compromised']}")

    print("\n=== Summary (trained) ===")
    print(f"  mean reward: {np.mean(rewards):+.2f}  (std {np.std(rewards):.2f})")
    print(f"  mean length: {np.mean(lengths):.1f}")
    print(f"  mean compromised at end: {np.mean(compromised):.2f}")

    # ---- action distribution check (collapse detector) ----
    print("\n=== Action distribution (trained) ===")
    counts = Counter(all_actions)
    total_acts = len(all_actions)
    for a in sorted(counts.keys()):
        pct = 100 * counts[a] / total_acts
        print(f"  action {a}: {counts[a]:4d}  ({pct:5.1f}%)")
    n_unique = len(counts)
    print(f"  unique actions used: {n_unique} / 5")
    if n_unique <= 1:
        print("  !!! POLICY COLLAPSED: only one action chosen !!!")
    elif n_unique == 2:
        print("  WARNING: policy uses only 2 actions")

    # ---- random baseline ----
    print(f"\n=== Random baseline: {N_EPISODES} episodes ===")
    rand_rewards, rand_compromised = [], []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=1000 + ep)     # SAME seeds as trained
        total = 0.0
        for t in range(env.max_steps):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        rand_rewards.append(total)
        rand_compromised.append(info["n_compromised"])

    print(f"  mean reward: {np.mean(rand_rewards):+.2f}  (std {np.std(rand_rewards):.2f})")
    print(f"  mean compromised at end: {np.mean(rand_compromised):.2f}")

    # ---- verdict ----
    gap = np.mean(rewards) - np.mean(rand_rewards)
    print(f"\n=== VERDICT ===")
    print(f"  Trained: {np.mean(rewards):+.2f}")
    print(f"  Random:  {np.mean(rand_rewards):+.2f}")
    print(f"  Gap:     {gap:+.2f}")
    if gap > 20:
        print("  OK -- trained agent clearly beats random.")
    elif gap > 5:
        print("  MARGINAL -- trained beats random by a small margin.")
    elif gap > -5:
        print("  NEUTRAL -- trained is roughly equal to random.")
    else:
        print("  BROKEN -- trained is worse than random (policy collapsed).")

    env.close()


if __name__ == "__main__":
    main()

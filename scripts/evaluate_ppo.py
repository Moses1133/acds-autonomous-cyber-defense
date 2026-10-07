"""
Evaluate a trained PPO model on CyberDefenseEnv (v5 or v6).

Usage:
    python scripts\evaluate_ppo.py                # eval v5
    python scripts\evaluate_ppo.py --env v6       # eval v6
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
import numpy as np
from stable_baselines3 import PPO


N_EPISODES = 20


def make_env(version):
    if version == "v5":
        from src.sim.environment import CyberDefenseEnv
        return CyberDefenseEnv(n_hosts=10, max_steps=100)
    elif version in ("v6", "v7"):
        from src.sim.environment_v6 import CyberDefenseEnvV6
        return CyberDefenseEnvV6(n_hosts=10, max_steps=100)
    elif version == "v7":
        from src.sim.environment_v7 import CyberDefenseEnvV7
        return CyberDefenseEnvV7(n_hosts=10, max_steps=100)
    raise ValueError(f"Unknown env version: {version}")


def flatten_action(a, version):
    """Convert env-specific action to a canonical int for reporting."""
    if version in ("v6", "v7"):
        return int(a[0])   # only action type
    return int(a)


def run_episode(model, env, seed, version):
    obs, info = env.reset(seed=seed)
    total = 0.0
    steps = 0
    actions = []
    for t in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        actions.append(flatten_action(a, version))
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps = t + 1
        if term or trunc:
            break
    return total, steps, info, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="v5", choices=["v5", "v6", "v7"])
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--model", default=None,
                        help="Path to model .zip. Defaults to models/ppo_acds_<env>.zip")
    args = parser.parse_args()

    version = args.env
    model_name = f"ppo_acds_{version}"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Find model
    # Find model -- honor --model if given
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"--model path does not exist: {model_path}")
    else:
        model_path = os.path.join(root, "models", f"{model_name}.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(root, "models", f"{model_name}_final.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(root, "models", f"{model_name}_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for {version}. "
            f"Run `python scripts\\train_ppo.py --env {version}` first."
        )

    print(f"=== Evaluating {version} model ===")
    print(f"Loading: {model_path}")
    model = PPO.load(model_path)
    env = make_env(version)

    print(f"\n=== Trained agent: {args.episodes} episodes ===")
    rewards, lengths, compromised, all_actions = [], [], [], []
    for ep in range(args.episodes):
        r, length, info, acts = run_episode(model, env, seed=1000 + ep,
                                            version=version)
        rewards.append(r); lengths.append(length)
        compromised.append(info["n_compromised"])
        all_actions.extend(acts)
        if ep < 5:
            print(f"  episode {ep+1:2d}: reward={r:+7.2f}  length={length:3d}  "
                  f"compromised={info['n_compromised']}")

    print(f"\n=== Summary (trained, {version}) ===")
    print(f"  mean reward:      {np.mean(rewards):+.2f}  (std {np.std(rewards):.2f})")
    print(f"  mean length:      {np.mean(lengths):.1f}")
    print(f"  mean compromised: {np.mean(compromised):.2f}")

    # Action distribution (only meaningful for v5; for v6 we flatten to action type)
    print(f"\n=== Action type distribution ===")
    counts = Counter(all_actions)
    total_acts = len(all_actions)
    for a in sorted(counts.keys()):
        pct = 100 * counts[a] / total_acts
        print(f"  action {a}: {counts[a]:5d}  ({pct:5.1f}%)")
    n_unique = len(counts)
    print(f"  unique action types used: {n_unique}")

    # Random baseline with the SAME seeds
    print(f"\n=== Random baseline: {args.episodes} episodes ===")
    rand_rewards, rand_compromised = [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=1000 + ep)
        total = 0.0
        for t in range(env.max_steps):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        rand_rewards.append(total)
        rand_compromised.append(info["n_compromised"])

    print(f"  mean reward:      {np.mean(rand_rewards):+.2f}")
    print(f"  mean compromised: {np.mean(rand_compromised):.2f}")

    gap = np.mean(rewards) - np.mean(rand_rewards)
    print(f"\n=== VERDICT ({version}) ===")
    print(f"  Trained: {np.mean(rewards):+.2f}")
    print(f"  Random:  {np.mean(rand_rewards):+.2f}")
    print(f"  Gap:     {gap:+.2f}")
    if gap > 20:
        print("  OK - trained clearly beats random.")
    elif gap > 5:
        print("  MARGINAL - trained beats random by a small margin.")
    elif gap > -5:
        print("  NEUTRAL - trained roughly equals random.")
    else:
        print("  BROKEN - trained worse than random.")

    env.close()


if __name__ == "__main__":
    main()




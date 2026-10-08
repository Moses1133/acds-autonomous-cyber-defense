"""
Baseline heuristics for ACDS - non-learning defenders.

Run: python scripts\baseline_heuristics.py --env v7

Heuristics evaluated:
  - noop          : do nothing every step (lower bound)
  - random        : sample a random (action, target) pair
  - isolate_worst : isolate the host with the lowest health
  - block_worst   : close one port on the lowest-health host
  - patch_most_vuln : patch a CVE on the host with the most CVEs
  - round_robin   : cycle through hosts, isolating each in turn

Compare results against the trained PPO model:
  python scripts\evaluate_ppo.py --env v7
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np


def make_env(version):
    if version == "v5":
        from src.sim.environment import CyberDefenseEnv
        return CyberDefenseEnv(n_hosts=10, max_steps=100)
    elif version == "v6":
        from src.sim.environment_v6 import CyberDefenseEnvV6
        return CyberDefenseEnvV6(n_hosts=10, max_steps=100)
    elif version == "v7":
        from src.sim.environment_v7 import CyberDefenseEnvV7
        return CyberDefenseEnvV7(n_hosts=10, max_steps=100)
    raise ValueError(f"Unknown env version: {version}")


def _lowest_health_host(env):
    hosts = [h for h in env.net.hosts.values() if h.role != "router"]
    if not hosts:
        return 0
    return min(hosts, key=lambda h: (h.health, h.id)).id


def _most_vuln_host(env):
    hosts = [h for h in env.net.hosts.values() if h.role != "router"]
    if not hosts:
        return 0
    return max(hosts, key=lambda h: (len(h.vulnerabilities), -h.health)).id


# ---------- heuristic policies ----------

def policy_noop(env, t, rng):
    return [0, 0]   # noop


def policy_random(env, t, rng):
    return [int(rng.integers(0, 5)), int(rng.integers(0, env.n_hosts))]


def policy_isolate_worst(env, t, rng):
    return [3, _lowest_health_host(env)]   # 3 = isolate


def policy_block_worst(env, t, rng):
    return [2, _lowest_health_host(env)]   # 2 = block_port


def policy_patch_most_vuln(env, t, rng):
    return [1, _most_vuln_host(env)]       # 1 = patch


def policy_round_robin(env, t, rng):
    return [3, (t % env.n_hosts)]          # isolate host `t % n_hosts`


HEURISTICS = {
    "noop":             policy_noop,
    "random":           policy_random,
    "isolate_worst":    policy_isolate_worst,
    "block_worst":      policy_block_worst,
    "patch_most_vuln":  policy_patch_most_vuln,
    "round_robin":      policy_round_robin,
}


def run_policy(policy_fn, env, seed, n_episodes=20):
    rewards, compromised = [], []
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        total = 0.0
        for t in range(env.max_steps):
            a = policy_fn(env, t, rng)
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        rewards.append(total)
        compromised.append(info["n_compromised"])
    return float(np.mean(rewards)), float(np.mean(compromised))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="v7", choices=["v5", "v6", "v7"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = make_env(args.env)

    print(f"=== Baseline heuristics on {args.env} ({args.episodes} episodes each) ===\n")
    print(f"{'policy':20s}  {'mean reward':>12s}  {'std':>7s}  {'compromised':>12s}")
    print("-" * 60)

    results = {}
    for name, fn in HEURISTICS.items():
        rng = np.random.default_rng(args.seed)
        rewards, comp = [], []
        for ep in range(args.episodes):
            obs, info = env.reset(seed=1000 + ep)
            total = 0.0
            for t in range(env.max_steps):
                a = fn(env, t, rng)
                obs, r, term, trunc, info = env.step(a)
                total += r
                if term or trunc:
                    break
            rewards.append(total)
            comp.append(info["n_compromised"])
        mean_r = float(np.mean(rewards))
        std_r  = float(np.std(rewards))
        mean_c = float(np.mean(comp))
        results[name] = (mean_r, mean_c)
        print(f"{name:20s}  {mean_r:>+12.2f}  {std_r:>7.2f}  {mean_c:>12.2f}")

    print()
    print("=== Comparison table ===")
    print(f"{'policy':20s}  {'reward':>10s}  {'compromised':>12s}")
    print("-" * 48)
    for name, (r, c) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{name:20s}  {r:>+10.2f}  {c:>12.2f}")

    print()
    print("To compare against PPO: python scripts\\evaluate_ppo.py --env v7")
    env.close()


if __name__ == "__main__":
    main()

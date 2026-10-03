"""
Train a PPO agent on CyberDefenseEnv.

Usage:
    python scripts\train_ppo.py                 # trains on v5 (default)
    python scripts\train_ppo.py --env v6        # trains on v6 (MultiDiscrete)
    python scripts\train_ppo.py --env v6 --steps 300000

Outputs (v5):
    models\ppo_acds_v5.zip         (best checkpoint)
    models\ppo_acds_v5_final.zip   (final checkpoint)

Outputs (v6):
    models\ppo_acds_v6.zip         (best checkpoint)
    models\ppo_acds_v6_final.zip   (final checkpoint)
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


def make_env(version: str):
    if version == "v5":
        from src.sim.environment import CyberDefenseEnv
        return CyberDefenseEnv(n_hosts=10, max_steps=100)
    elif version == "v6":
        from src.sim.environment_v6 import CyberDefenseEnvV6
        return CyberDefenseEnvV6(n_hosts=10, max_steps=100)
    elif version == "v7":
        from src.sim.environment_v7 import CyberDefenseEnvV7
        return CyberDefenseEnvV7(n_hosts=10, max_steps=100)
    else:
        raise ValueError(f"Unknown env version: {version}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="v5", choices=["v5", "v6", "v7"],
                        help="Environment version to train on")
    parser.add_argument("--steps", type=int, default=300_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    version = args.env
    model_name = f"ppo_acds_{version}"
    total_timesteps = args.steps

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(root, "models", model_name)
    log_dir = os.path.join(root, "logs", version)

    # Clean stale best-model file
    stale_best = os.path.join(root, "models", f"{model_name}.zip")
    if os.path.exists(stale_best):
        os.remove(stale_best)
        print(f"Removed stale {stale_best}")

    print(f"=== Building {version} envs ===")
    train_env = Monitor(make_env(version))
    eval_env = Monitor(make_env(version))

    print(f"  obs space: {train_env.observation_space}")
    print(f"  act space: {train_env.action_space}")

    print(f"\n=== Building PPO model ({version}) ===")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=4096,
        batch_size=128,
        gamma=0.99,
        learning_rate=1e-4,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log=log_dir,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_dir,
        eval_freq=15_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    print("=" * 60)
    print(f"Training PPO [{version}] for {total_timesteps:,} timesteps")
    print(f"Model output: {model_path}.zip")
    print(f"TensorBoard logs: {log_dir}")
    print("=" * 60)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb,
                progress_bar=False)
    dt = time.time() - t0

    final_path = model_path + "_final"
    model.save(final_path)
    print(f"\nDone in {dt:.1f}s")
    print(f"Final model saved: {final_path}.zip")
    print(f"Best model saved:  {model_path}.zip")


if __name__ == "__main__":
    main()


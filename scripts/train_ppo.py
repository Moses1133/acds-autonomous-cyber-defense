"""
Train a PPO agent on CyberDefenseEnv.

Run: python scripts\train_ppo.py

Outputs:
- models\ppo_acds_v1.zip            (trained policy)
- logs\                             (TensorBoard event files)

View training curves with:
  tensorboard --logdir logs
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.sim.environment import CyberDefenseEnv


TOTAL_TIMESTEPS = 200_000
SEED = 42


def main():
    # --- make paths stable regardless of where we run from ---
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(root, "models", "ppo_acds_v1")
    log_dir = os.path.join(root, "logs")

    # --- build envs ---
    train_env = Monitor(CyberDefenseEnv(n_hosts=10, max_steps=100))
    eval_env  = Monitor(CyberDefenseEnv(n_hosts=10, max_steps=100))

    # --- build model ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        verbose=1,
        seed=SEED,
        tensorboard_log=log_dir,
    )

    # --- eval callback: saves the best model along the way ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print("=" * 60)
    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"Model output: {model_path}.zip")
    print(f"TensorBoard logs: {log_dir}")
    print("=" * 60)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb, progress_bar=False)
    dt = time.time() - t0

    # Final save
    final_path = model_path + "_final"
    model.save(final_path)
    print(f"\nDone in {dt:.1f}s")
    print(f"Final model saved: {final_path}.zip")
    print(f"Best model saved during training: {model_path}.zip")


if __name__ == "__main__":
    main()

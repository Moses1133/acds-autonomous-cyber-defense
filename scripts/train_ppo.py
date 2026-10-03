"""
Train a PPO agent on CyberDefenseEnv.

Fixes applied (Day 9 - policy collapse prevention):
- entropy_coef=0.01   (was 0.0) -- keeps policy stochastic
- learning_rate=1e-4  (was 3e-4) -- slower, more stable
- n_steps=4096        (was 2048) -- lower-variance gradients

Run: python scripts\train_ppo.py

Outputs:
- models\ppo_acds_v1.zip        (best checkpoint during training)
- models\ppo_acds_v1_final.zip  (final checkpoint)
- logs\                         (TensorBoard event files)

View with: tensorboard --logdir logs
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
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(root, "models", "ppo_acds_v1")
    log_dir = os.path.join(root, "logs")

    # --- clean up any stale best_model from a previous run ---
    stale_best = os.path.join(root, "models", "ppo_acds_v1.zip")
    if os.path.exists(stale_best):
        os.remove(stale_best)
        print(f"Removed stale {stale_best}")

    # --- envs ---
    train_env = Monitor(CyberDefenseEnv(n_hosts=10, max_steps=100))
    eval_env  = Monitor(CyberDefenseEnv(n_hosts=10, max_steps=100))

    # --- model with collapse-prevention hyperparams ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=4096,                 # was 2048
        batch_size=128,               # scaled with n_steps
        gamma=0.99,
        learning_rate=1e-4,           # was 3e-4
        ent_coef=0.01,                # was default 0.0 -- PREVENTS COLLAPSE
        verbose=1,
        seed=SEED,
        tensorboard_log=log_dir,
    )

    # --- eval callback saves best checkpoint ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=10,           # was 5
        deterministic=True,
        render=False,
    )

    print("=" * 60)
    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"Model output: {model_path}.zip")
    print(f"TensorBoard logs: {log_dir}")
    print("Hyperparameters: entropy_coef=0.01, lr=1e-4, n_steps=4096")
    print("=" * 60)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb, progress_bar=False)
    dt = time.time() - t0

    final_path = model_path + "_final"
    model.save(final_path)
    print(f"\nDone in {dt:.1f}s")
    print(f"Final model saved: {final_path}.zip")
    print(f"Best model saved during training: {model_path}.zip")


if __name__ == "__main__":
    main()

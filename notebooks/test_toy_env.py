"""
Sanity-check the toy env - verify the Gymnasium API contract.
Run: python notebooks\test_toy_env.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from notebooks.toy_env import ToyGridEnv

env = ToyGridEnv()

obs, info = env.reset()
print(f"Start: obs={obs}, info={info}")

total_reward = 0
for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"step={step:2d}  action={action}  obs={obs}  reward={reward:+.2f}  "
          f"term={terminated}  trunc={truncated}")
    if terminated or truncated:
        print(f"Episode done at step {step}. Total reward: {total_reward:.2f}")
        break

env.close()
print("\nToy env works.")

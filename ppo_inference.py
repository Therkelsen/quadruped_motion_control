import time
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from ppo import MyCustomEnv  # <-- your training script filename

if __name__ == "__main__":
    # Create a function to instantiate the environment
    def make_env():
        env = MyCustomEnv(human_friendly=True)  # GUI mode
        env = Monitor(env)  # optional: monitor for per-episode rewards
        return env

    env = DummyVecEnv([make_env])

    # Load the saved VecNormalize stats
    env = VecNormalize.load("./models/vecnormalize_latest.pkl", env)
    env.training = False  # Important! Don't update running stats during inference
    env.norm_reward = False  # Optional: disable reward normalization for viewing raw rewards

    # Load the trained model
    model = PPO.load("./models/ppo_go2_latest", env=env)

    obs, _ = env.reset()
    done = False

    while not done:
        # Model predicts action
        action, _ = model.predict(obs, deterministic=True)

        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Optional: print info
        print(f"Step reward: {reward}, done: {done}")

        # Slow down for GUI visualization
        time.sleep(1./240)

    env.close()

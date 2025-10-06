import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import your environment class
from ppo import MyCustomEnv  # <-- change to the filename of your script

if __name__ == "__main__":
    # Create environment in GUI mode for visualization
    env = MyCustomEnv(human_friendly=True)

    # Optional: check if the environment follows Gym API
    check_env(env, warn=True)

    # Load trained PPO model
    model_path = "./models/ppo_go2_5000_steps.zip"  # <-- path to your saved model
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()

    done = False
    while not done:
        # Model predicts action given observation
        action, _ = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Stop if episode ends
        done = terminated or truncated

        # Optional: print some info
        #print(f"Step Reward: {reward}, Action: {action}")

        # Slow down simulation to see it
        time.sleep(1./240)

    env.close()

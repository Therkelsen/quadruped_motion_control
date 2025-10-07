import time
import numpy as np
import pybullet as p
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

# Import your environment class
from sac import MyCustomEnv  # <-- update if your environment is in another file name

if __name__ == "__main__":
    # Create environment in GUI mode
    env = MyCustomEnv(human_friendly=True)

    # Optional: check environment API compliance
    check_env(env, warn=True)

    # Load trained SAC model
    model_path = "./models/sac_go2_50000_steps.zip"  # <-- change to your saved model
    model = SAC.load(model_path, env=env)

    # Reset environment
    obs, _ = env.reset()
    done = False

    print("Starting SAC inference...")

    while not done:
        # Predict deterministic action
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # (Optional) print debug info
        # print(f"Reward: {reward:.3f}")

        # Slow down for visualization
        time.sleep(1./240)

    env.close()
    print("Inference finished.")

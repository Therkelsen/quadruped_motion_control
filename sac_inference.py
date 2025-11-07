import time
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Import your environment class
from sac import MyCustomEnv  # replace 'sac' with your actual training file name

if __name__ == "__main__":
    # Function to create a new environment
    def make_env():
        env = MyCustomEnv(human_friendly=True)  # GUI mode
        check_env(env, warn=True)  # optional, only needed once
        return env

    # Wrap in DummyVecEnv
    env = DummyVecEnv([make_env])

    # Load the trained SAC model
    model_path = "./models/sac_go2_500000_steps.zip"  # change to your saved model
    model = SAC.load(model_path, env=env)

    obs = env.reset()
    print("Starting SAC inference...")

    # Run indefinitely for visualization
    while True:
        # Predict next action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, done, info = env.step(action)

        # VecEnv returns arrays, convert to scalar
        done = np.any(done)

        # Slow down for GUI visualization
        time.sleep(1./240)

        # Auto-reset when episode ends
        if done:
            obs = env.reset()

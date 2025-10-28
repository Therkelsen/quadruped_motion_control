import time
import numpy as np
import pybullet as p
import pybullet_data

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from ppo import MyCustomEnv  # Import your environment class


if __name__ == "__main__":
    # Create a GUI environment for visualization
    def make_env():
        env = MyCustomEnv(human_friendly=True)  # human_friendly=True = PyBullet GUI
        env = Monitor(env)  # Optional, logs episode rewards and lengths
        return env

    env = DummyVecEnv([make_env])  # PPO expects a vectorized env, even if only 1

    # Load the trained PPO model
    model = PPO.load("./models/ppo_go2_latest.zip", env=env)
    print("✅ Loaded trained model successfully.")

    obs = env.reset()
    done = False

    # Main simulation loop
    while True:
        # Predict next action from the model
        action, _ = model.predict(obs, deterministic=True)  # deterministic=True for stable playback

        # Step environment
        obs, reward, done, info = env.step(action)
        done = np.any(done)  # VecEnv returns array, so convert to bool

        # Optional: print reward
        # print(f"Reward: {reward}")

        # Slow down to real time for GUI visualization
        time.sleep(1.0 / 240.0)

        # Reset if episode ended
        if done:
            obs = env.reset()

    env.close()

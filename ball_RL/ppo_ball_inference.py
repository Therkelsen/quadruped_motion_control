import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from ppo_ball_train import BallEnv


if __name__ == "__main__":

    env = BallEnv(human_friendly=True)
    model = PPO.load("./models/ppo_ball_latest")

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(1 / 240)
        
        if terminated or truncated:
            obs, _ = env.reset()

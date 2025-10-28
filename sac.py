from ppo import MyCustomEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import array
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


if __name__ == "__main__":
    def make_env():
        env = MyCustomEnv(human_friendly=False)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    log_dir = "./tensorboard/"
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path='./models/', name_prefix='sac_go2')

    # Use SAC instead of PPO
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=500_000,    # smaller buffer
        batch_size=128,         # smaller batch
        learning_starts=10_000,
        tau=0.005,
        gamma=0.99,
        train_freq=1,           # every step
        gradient_steps=1,       # only 1 update per step
        ent_coef='auto'
    )

    
    total_timesteps = 1_000_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    env.close()
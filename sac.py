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


if __name__ == "__main__":
    env = MyCustomEnv(human_friendly=False)

    log_dir = "./tensorboard/"
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/', name_prefix='sac_go2')

    # Use SAC instead of PPO
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,     # typical default for SAC
        buffer_size=1000000,    # replay buffer size
        batch_size=256,         # SAC benefits from larger batch size
        learning_starts=10000,  # number of steps before learning starts
        tau=0.005,              # target smoothing coefficient
        gamma=0.99,             # discount factor
        train_freq=1,           # train every step
        gradient_steps=1,       # one gradient update per step
        ent_coef='auto'         # automatic entropy tuning
    )

    model.learn(total_timesteps=100000000, callback=checkpoint_callback)

    env.close()
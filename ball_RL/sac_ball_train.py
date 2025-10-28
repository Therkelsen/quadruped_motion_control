from ppo_ball_train import BallEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import array
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import time


if __name__ == "__main__":
    def make_env():
        env = BallEnv(human_friendly=False)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = BallEnv(human_friendly=False)

    log_dir = "./tensorboard/"
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='sac_ball')

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

    total_timesteps = 1_000_000
    
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/sac_ball_latest")

    env.close()
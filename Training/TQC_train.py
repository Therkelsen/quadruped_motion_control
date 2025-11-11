# train_tqc.py
import os
import argparse
import pickle
import shutil
import numpy as np

import genesis as gs
from sb3_contrib import TQC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="TQC")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # ---------------- Training parameters ----------------
    num_envs = 16   # Vectorized environments
    replay_buffer_size = 3_000_000
    batch_size = 512
    learning_starts = 30_000
    tau = 0.005
    train_freq = 128
    gradient_steps = 128
    gamma = 0.99
    learning_rate = 3e-4
    sigma = 0.1  # Action noise

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- Load configs ----------------
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)

    # Save configs for reproducibility
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # ---------------- Create Genesis-based VecEnv ----------------
    vec_env = GenesisVecEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # ---------------- Normalize observations & rewards ----------------
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---------------- Action noise ----------------
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))

    # ---------------- Define TQC ----------------
    # Define TQC (fixed)
    model = TQC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        buffer_size=replay_buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        action_noise=action_noise,
        n_quantiles=25,                  # number of quantiles
        top_quantiles_to_drop=2,         # how many top quantiles to drop per critic
    )


    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)

    # ---------------- Save model & VecNormalize stats ----------------
    model.save(os.path.join(log_dir, "tqc"))
    vec_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"✅ Model saved at {log_dir}/tqc.zip")
    print(f"✅ VecNormalize stats saved at {log_dir}/vecnormalize.pkl")

    vec_env.close()


if __name__ == "__main__":
    main()

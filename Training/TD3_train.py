import os
import argparse
import pickle
import shutil
import numpy as np

import genesis as gs
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="TD3")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # ---------------- Environment + training setup ----------------
    num_envs = 16  # or 32 if memory allows
    replay_buffer_size = 3_000_000
    batch_size = 512  # or 1024 if GPU memory allows
    learning_starts = 30_000
    tau = 0.005
    train_freq = 128
    gradient_steps = 128


    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- Load configs ----------------
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)

    env_cfg["action_scale"] = 0.35   # strong enough to move but not chaotic
    env_cfg["kp"] = 60.0
    env_cfg["kd"] = 1.0
    env_cfg["simulate_action_latency"] = False
    
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
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # ---------------- Define TD3 ----------------
    model = TD3(
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
        gamma=train_cfg["algorithm"].get("gamma", 0.99),
        learning_rate=train_cfg["algorithm"].get("learning_rate", 3e-4),
        action_noise=action_noise,
    )

    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)

    # ---------------- Save model & VecNormalize stats ----------------
    model.save(os.path.join(log_dir, "td3"))
    vec_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"✅ Model saved at {log_dir}/td3.zip")
    print(f"✅ VecNormalize stats saved at {log_dir}/vecnormalize.pkl")

    vec_env.close()


if __name__ == "__main__":
    main()

# train_td3.py
import os
import argparse
import pickle
import shutil
import numpy as np

import genesis as gs
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="TD3")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    num_envs = 1024
    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ✅ Load configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)

    # Save configs
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # ✅ Create Genesis-based VecEnv
    vec_env = GenesisVecEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # ✅ Action noise for exploration (important for TD3)
    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # ✅ Define TD3
    model = TD3(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        action_noise=action_noise,
        learning_rate=train_cfg["algorithm"]["learning_rate"],
        gamma=train_cfg["algorithm"]["gamma"],
        buffer_size=1_000_000,
        batch_size=256,
    )

    # ✅ Train
    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(log_dir, "td3"))
    print(f"✅ Model saved at {log_dir}/td3.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

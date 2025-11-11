# ARS_train.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from sb3_contrib import ARS
from stable_baselines3.common.vec_env import VecMonitor
from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ARS")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # ---------------- Training parameters ----------------
    num_envs = 64  # ARS supports vectorized envs, but very high numbers may be slow
    n_steps = 1    # ARS uses one-step rollouts per perturbation
    sigma = 0.05   # noise standard deviation
    learning_rate = 0.02  # ARS learning rate
    n_top_directions = 16
    n_directions = 32

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

    # ---------------- Wrap in VecMonitor for ARS evaluation ----------------
    vec_env = VecMonitor(vec_env)

    # ---------------- Define ARS ----------------
    model = ARS(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=n_steps,
        sigma=sigma,
        learning_rate=learning_rate,
        n_directions=n_directions,
        n_top_directions=n_top_directions,
        tensorboard_log=log_dir,
    )

    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)

    # ---------------- Save model ----------------
    model.save(os.path.join(log_dir, "ars"))
    print(f"✅ Model saved at {log_dir}/ars.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

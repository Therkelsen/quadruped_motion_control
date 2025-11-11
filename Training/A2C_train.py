# train_a2c.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import A2C  # Or ACKTR if you want
from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="A2C")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # ---------------- Training parameters ----------------
    num_envs = 1024  # Vectorized environments
    n_steps = 24     # Number of steps per environment before update
    gamma = 0.99
    learning_rate = 3e-4
    ent_coef = 0.01

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

    # ---------------- Create Genesis VecEnv ----------------
    vec_env = GenesisVecEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # ---------------- Define A2C ----------------
    model = A2C(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=n_steps,
        gamma=train_cfg["algorithm"].get("gamma", gamma),
        learning_rate=train_cfg["algorithm"].get("learning_rate", learning_rate),
        ent_coef=train_cfg["algorithm"].get("entropy_coef", ent_coef),
    )

    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)

    # ---------------- Save model ----------------
    model.save(os.path.join(log_dir, "a2c"))
    print(f"✅ Model saved at {log_dir}/a2c.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

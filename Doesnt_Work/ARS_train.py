# ARS_train.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from sb3_contrib.ars import ARS
from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ARS")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    num_envs = 16  # ARS works with vectorized envs, keep reasonable
    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Load configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)

    # Save configs
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Genesis VecEnv
    vec_env = GenesisVecEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # Define ARS
    model = ARS(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        # device="cpu"  # ARS is CPU-based; uncomment if you want to force CPU
    )

    # Train
    model.learn(total_timesteps=args.total_timesteps)

    # Save
    model.save(os.path.join(log_dir, "ars"))
    print(f"✅ Model saved at {log_dir}/ars.zip")

    vec_env.close()

if __name__ == "__main__":
    main()

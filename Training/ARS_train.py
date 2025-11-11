# train_ars.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from sb3_contrib import ARS
from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ARS")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    num_envs = 1024  # vectorized environments for parallel rollouts

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

    # ---------------- Define ARS ----------------
    model = ARS(
        policy="MlpPolicy",       # small MLP or "LinearPolicy"
        env=vec_env,
        n_directions=16,          # number of perturbation directions per iteration
        n_top_directions=8,       # best directions used to update policy
        noise_std=0.03,           # perturbation noise
        learning_rate=0.02,       # policy update step
        verbose=1,
        seed=42,
    )

    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)

    # ---------------- Save model ----------------
    model.save(os.path.join(log_dir, "ars"))
    print(f"✅ Model saved at {log_dir}/ars.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

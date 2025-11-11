# RecurrentPPO_train.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from sb3_contrib import RecurrentPPO
from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="RecurrentPPO")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    num_envs = 64  # vectorized environments
    n_steps = 128  # number of steps per environment
    batch_size = 256

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

    # ---------------- Define RecurrentPPO ----------------
    model = RecurrentPPO(
        "MlpLstmPolicy",  # Recurrent policy
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=train_cfg["algorithm"]["learning_rate"],
        gamma=train_cfg["algorithm"]["gamma"],
        ent_coef=train_cfg["algorithm"]["entropy_coef"],
    )

    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(log_dir, "recurrent_ppo"))
    print(f"✅ Model saved at {log_dir}/recurrent_ppo.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

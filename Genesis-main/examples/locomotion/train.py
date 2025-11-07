# train.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import PPO

from Gymwrapper import GenesisVecEnv  # the custom VecEnv above
from go2_env import get_cfgs  # function to get env configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-sb3")
    parser.add_argument("-B", "--num_envs", type=int, default=1024, help="Number of parallel envs inside Genesis (GPU).")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO n_steps per rollout (per env).")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_cfgs(args.exp_name, max_iterations=100)  # keep for saving

    # Save cfgs (like the rsl-rl script does)
    pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))

    # Create GenesisVecEnv with many environments simulated inside GPU
    vec_env = GenesisVecEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # Choose PPO and tune n_steps according to your setup.
    # n_steps in SB3 is number of steps per environment collected per update.
    # total batch size per update = n_steps * num_envs
    # Keep batch sizes reasonable for memory: e.g., n_steps=24 (like rsl-rl) gives batch = 24*num_envs.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=args.n_steps,
        batch_size=256,  # minibatch size for PPO; adjust to taste (<= n_steps * num_envs)
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
    )

    # Train
    model.learn(total_timesteps=args.total_timesteps)

    # Save model
    model.save(os.path.join(log_dir, "ppo"))
    print("Model saved to:", os.path.join(log_dir, "ppo.zip"))

    vec_env.close()


if __name__ == "__main__":
    main()

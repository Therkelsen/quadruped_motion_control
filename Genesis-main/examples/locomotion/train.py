# train.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import PPO

from Gymwrapper import GenesisVecEnv
from go2_train import get_cfgs, get_train_cfg  # ✅ imported from your existing Genesis RSL-RL script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-sb3")
    parser.add_argument("-B", "--num_envs", type=int, default=1024,
                        help="Number of parallel envs simulated on GPU in Genesis.")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_steps", type=int, default=24,
                        help="Steps per environment per PPO rollout (like RSL-RL).")
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ✅ Load configs from original go2_train.py
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)

    # Save configs for reproducibility (like RSL-RL)
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # ✅ Create Genesis-based VecEnv (GPU vectorized)
    vec_env = GenesisVecEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # ✅ Define PPO (SB3)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=train_cfg["algorithm"]["learning_rate"],
        gamma=train_cfg["algorithm"]["gamma"],
        ent_coef=train_cfg["algorithm"]["entropy_coef"],
    )

    # ✅ Train
    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(log_dir, "ppo"))
    print(f"✅ Model saved at {log_dir}/ppo.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

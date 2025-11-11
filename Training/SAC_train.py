import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import SAC

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-sb3")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # ---------------- Environment + training setup ----------------
    num_envs = 32
    replay_buffer_size = 2_000_000
    sac_batch_size = 512
    learning_starts = 20_000
    tau = 0.005
    train_freq = 64
    gradient_steps = 64

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- Load configs ----------------
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)

    # Reward scales
    reward_cfg["reward_scales"] = {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 1.0,
        "lin_vel_z": -0.1,
        "base_height": -1.0,
        "action_rate": -0.01,
        "similar_to_default": -0.05,
    }

    # Save configs
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

    # ---------------- Define SAC ----------------
    ent_coef = train_cfg["algorithm"].get("entropy_coef", "auto")
    learning_rate = train_cfg["algorithm"].get("learning_rate", 3e-4)
    gamma = train_cfg["algorithm"].get("gamma", 0.99)
    target_entropy = -0.5 * vec_env.action_space.shape[0]

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        buffer_size=replay_buffer_size,
        batch_size=sac_batch_size,
        learning_starts=learning_starts,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
    )

    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)

    # ---------------- Save model ----------------
    model.save(os.path.join(log_dir, "sac"))
    print(f"✅ Model saved at {log_dir}/sac.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

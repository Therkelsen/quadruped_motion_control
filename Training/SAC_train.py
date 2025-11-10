# train.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-sb3")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    
    args = parser.parse_args()
    
    num_envs = 4
    # SAC (off-policy) hyperparameters
    replay_buffer_size = 1_000_000
    sac_batch_size = 256
    learning_starts = 10_000
    tau = 0.005
    train_freq = 64
    gradient_steps = 64
    
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
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # ✅ Define SAC (SB3)
    ent_coef = train_cfg["algorithm"].get("entropy_coef", "auto_1")
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        buffer_size=replay_buffer_size,
        batch_size=sac_batch_size,
        learning_starts=learning_starts,
        learning_rate=train_cfg["algorithm"]["learning_rate"],
        gamma=train_cfg["algorithm"]["gamma"],
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
    )

    save_freq = 50_000
    checkpoint_callback = CheckpointCallback(
        save_freq = save_freq // num_envs,
        save_path = log_dir,
        name_prefix = "sac",
    )

    # ✅ Train
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    model.save(os.path.join(log_dir, "sac"))
    print(f"✅ Model saved at {log_dir}/sac.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

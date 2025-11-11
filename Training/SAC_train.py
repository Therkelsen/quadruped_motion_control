# SAC_train_fixed.py
import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg


# ---------------- Reward scaling wrapper ----------------
class RewardScaling(VecEnvWrapper):
    """Scale rewards from a VecEnv by a constant factor."""
    def __init__(self, venv, scale: float = 1.0):
        super().__init__(venv)
        self.scale = scale

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = rewards * self.scale
        return obs, rewards, dones, infos


# ---------------- Main training ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-sb3")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # Training hyperparameters
    num_envs = 32
    replay_buffer_size = 1_000_000
    sac_batch_size = 256
    learning_starts = 100_000
    tau = 0.005
    train_freq = 64
    gradient_steps = 64
    reward_scale_factor = 20.0  # scale small rewards

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Load configs
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

    # ---------------- Normalize observations & rewards ----------------
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Save VecNormalize stats immediately in case of early termination
    vec_env.save(os.path.join(log_dir, "vecnormalize.pkl"))

    # ---------------- Reward scaling wrapper ----------------
    vec_env = RewardScaling(vec_env, scale=reward_scale_factor)

    # ---------------- Define SAC model ----------------
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

    # ---------------- Save model and VecNormalize ----------------
    model.save(os.path.join(log_dir, "sac"))
    vec_env.venv.save(os.path.join(log_dir, "vecnormalize.pkl"))  # unwrap before saving
    print(f"✅ Model saved at {log_dir}/sac.zip")
    print(f"✅ VecNormalize stats saved at {log_dir}/vecnormalize.pkl")

    vec_env.close()


if __name__ == "__main__":
    main()

import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import SAC

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ActionLogger(BaseCallback):
    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.counter = 0

    def _on_step(self):
        self.counter += 1
        if self.counter % self.log_interval == 0:
            try:
                env = self.training_env
                while hasattr(env, 'envs'):
                    env = env.envs[0]
                if hasattr(env, 'env'):
                    env = env.env
                a = getattr(env, "actions", None)
                if a is not None:
                    a_np = a.detach().cpu().numpy()
                    print(f"[ActionLogger] step={self.counter} | mean={a_np.mean():.4f}, std={a_np.std():.4f}")
            except Exception as e:
                print(f"[ActionLogger] log error: {e}")
        return True



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="SAC")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # ---------------- Environment + training setup ----------------
    # Use more environments for better exploration and signal diversity
    num_envs = 64
    replay_buffer_size = 1_000_000
    sac_batch_size = 128
    learning_starts = 1000
    tau = 0.01
    train_freq = 1
    gradient_steps = 4

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- Load configs ----------------
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)
    
    # Test
    env_cfg["action_scale"] = 0.35   # strong enough to move but not chaotic
    env_cfg["kp"] = 60.0
    env_cfg["kd"] = 1.0
    env_cfg["simulate_action_latency"] = False

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

    # ---------------- Define SAC parameters ----------------
    ent_coef = train_cfg["algorithm"].get("entropy_coef", "auto")
    learning_rate = train_cfg["algorithm"].get("learning_rate", 3e-4)
    gamma = train_cfg["algorithm"].get("gamma", 0.99)
    target_entropy = -0.5 * vec_env.action_space.shape[0]
    
    #TEST
    target_entropy = "auto"
    
    # ========================================================================

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
    model.learn(total_timesteps=args.total_timesteps, callback=ActionLogger())

    # ---------------- Save model ----------------
    model.save(os.path.join(log_dir, "sac"))
    print(f"✅ Model saved at {log_dir}/sac.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

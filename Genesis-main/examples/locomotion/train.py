# go2_train_sb3.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Gymwrapper import Go2GymWrapper
from go2_env import Go2Env
from go2_train import get_cfgs  # or define it here if needed

def main():
    log_dir = "logs/go2-walking-sb3"
    os.makedirs(log_dir, exist_ok=True)

    # ✅ Get your environment configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # ✅ Pass configs into your wrapper
    env = make_vec_env(
        lambda: Go2GymWrapper(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg
        ),
        n_envs=8
    )

    # ✅ Use PPO (only)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,  # you can tune this
        batch_size=256,
    )

    model.learn(total_timesteps=2_000_000)
    model.save(os.path.join(log_dir, "ppo"))
    print("✅ Training complete and model saved.")


if __name__ == "__main__":
    main()

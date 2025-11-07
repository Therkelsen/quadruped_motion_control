# go2_train_sb3.py
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from Gymwrapper import Go2GymWrapper  # your wrapper above

def main():
    log_dir = "logs/go2-walking-sb3"
    os.makedirs(log_dir, exist_ok=True)

    # create wrapped environment (8 parallel environments)
    env = make_vec_env(lambda: Go2GymWrapper(), n_envs=8)

    # pick algorithm — choose ONE:
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
    )

    # train
    model.learn(total_timesteps=2_000_000)

    # save
    model.save(os.path.join(log_dir, "sac_go2"))

if __name__ == "__main__":
    main()

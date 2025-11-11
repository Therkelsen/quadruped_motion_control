import argparse
import os
import pickle

import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.Gymwrapper import Go2GymSingle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-sb3")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    gs.init()  # initialize Genesis

    log_dir = f"logs/{args.exp_name}"
    model_path = os.path.join(log_dir, "sac.zip")

    # ---------------- Load configs ----------------
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfgs_path):
        raise FileNotFoundError(f"cfgs.pkl not found in {log_dir}.")
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))

    # ---------------- Create single environment ----------------
    env_single = DummyVecEnv([lambda: Go2GymSingle(env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True)])

    # Load VecNormalize stats
    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        env_single = VecNormalize.load(vecnorm_path, env_single)
        env_single.training = False
        env_single.norm_reward = False  # don't normalize rewards at evaluation
    else:
        print("⚠️ No vecnormalize.pkl found — running without normalization!")

    # ---------------- Load SAC model ----------------
    model = SAC.load(model_path, env=env_single)

    # ---------------- Run evaluation ----------------
    for ep in range(args.episodes):
        obs = env_single.reset()
        done = [False]
        total_reward = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env_single.step(action)
            total_reward += reward[0]
        print(f"Episode {ep+1}/{args.episodes} — total_reward: {total_reward:.3f}")

    env_single.close()
    print("✅ Evaluation finished.")


if __name__ == "__main__":
    main()

# eval_tqc.py
import argparse
import os
import pickle
import genesis as gs
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from src.Gymwrapper import Go2GymSingle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="TQC")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_name = "tqc"

    gs.init()  # Initialize Genesis

    log_dir = f"logs/{args.exp_name}"
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load saved configs
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if os.path.exists(cfgs_path):
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))
    else:
        raise FileNotFoundError(f"cfgs.pkl not found in {log_dir}. Needed to reconstruct env config.")

    # Create single env
    env = Go2GymSingle(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Wrap in DummyVecEnv for VecNormalize compatibility
    env = DummyVecEnv([lambda: env])

    # Load VecNormalize stats if available
    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ No vecnormalize.pkl found — running without normalization!")

    # Load TQC model
    print(f"Loading TQC model from: {model_path}")
    model = TQC.load(model_path, env=env)

    # Run evaluation episodes
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep+1}/{args.episodes} — total_reward: {total_reward:.3f}")

    env.close()
    print("✅ Evaluation finished.")

if __name__ == "__main__":
    main()

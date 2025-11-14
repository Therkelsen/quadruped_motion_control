# ARS_eval.py
import argparse
import os
import pickle
import genesis as gs
from sb3_contrib.ars import ARS
from src.Gymwrapper import Go2GymSingle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ARS")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_name = "ars"
    gs.init()

    log_dir = f"logs/{args.exp_name}"
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load configs
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))

    # Load model
    model = ARS.load(model_path)

    # Single env with viewer
    env = Go2GymSingle(env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                       command_cfg=command_cfg, show_viewer=True)

    for ep in range(args.episodes):
        obs, _ = env.reset()
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

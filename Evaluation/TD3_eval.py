# eval_td3.py
import argparse
import os
import pickle
from stable_baselines3 import TD3
from src.Gymwrapper import Go2GymSingle
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="TD3")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_name = "td3"
        
    gs.init()  # Initialize Genesis (viewer etc.)

    log_dir = f"logs/{args.exp_name}"
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # ✅ Load saved configs for reproducibility
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if os.path.exists(cfgs_path):
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))
    else:
        raise FileNotFoundError(f"cfgs.pkl not found in {log_dir}. Needed to reconstruct env config.")

    print(f"Loading TD3 model: {model_path}")
    model = TD3.load(model_path)

    # ✅ Create single env with viewer
    env = Go2GymSingle(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # ✅ Run evaluation episodes
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # Deterministic policy for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}/{args.episodes} — total_reward: {total_reward:.3f}")

    env.env.close()
    print("✅ Evaluation finished.")


if __name__ == "__main__":
    main()

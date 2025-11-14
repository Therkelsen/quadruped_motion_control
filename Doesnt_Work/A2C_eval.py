# eval_a2c.py
import argparse
import os
import pickle
import genesis as gs
from stable_baselines3 import A2C  # Change to ACKTR if needed
from src.Gymwrapper import Go2GymSingle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="A2C")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_name = "a2c"  # Should match what you saved in training

    # ---------------- Initialize Genesis ----------------
    gs.init()  # Initialize Genesis (viewer, GPU, etc.)

    log_dir = f"logs/{args.exp_name}"
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # ---------------- Load training configs ----------------
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfgs_path):
        raise FileNotFoundError(f"cfgs.pkl not found in {log_dir}. Needed to reconstruct env config.")
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))

    # ---------------- Load model ----------------
    print(f"Loading A2C model: {model_path}")
    model = A2C.load(model_path)

    # ---------------- Create single env with viewer ----------------
    env = Go2GymSingle(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True
    )

    # ---------------- Run evaluation episodes ----------------
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

    env.env.close()  # Close Genesis environment properly
    print("✅ Evaluation finished.")

if __name__ == "__main__":
    main()

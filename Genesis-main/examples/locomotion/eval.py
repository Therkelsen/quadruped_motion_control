# go2_eval_sb3.py
import argparse
import os
from stable_baselines3 import PPO
from Gymwrapper import Go2GymWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_name", type=str, default="go2-walking-sb3",
        help="Experiment name / log folder"
    )
    parser.add_argument(
        "--model_name", type=str, default="ppo",
        help="Model filename (without .zip). Defaults to 'ppo'"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    model_path = os.path.join(log_dir, f"{args.model_name}.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    print(f"✅ Loading PPO model from {model_path}")
    model = PPO.load(model_path)

    # Create single-environment instance for evaluation
    env = Go2GymWrapper(num_envs=1)
    obs, _ = env.reset()

    print(f"🎯 Evaluating PPO model for {args.episodes} episodes...\n")

    for ep in range(args.episodes):
        done = False
        total_reward = 0.0
        obs, _ = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}/{args.episodes} | Total reward: {total_reward:.3f}")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()

"""
# Example usage:
python go2_eval_sb3.py -e go2-walking-sb3 --model_name ppo --episodes 3
"""

# RecurrentPPO_eval.py
import argparse
import os
import pickle
from sb3_contrib import RecurrentPPO
from src.Gymwrapper import Go2GymSingle
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="RECPPO")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_name = "RECPPO"

    gs.init()  # initialize genesis

    log_dir = f"logs/{args.exp_name}"
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load saved configs
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if os.path.exists(cfgs_path):
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))
    else:
        raise FileNotFoundError(f"cfgs.pkl not found in {log_dir}")

    print(f"Loading RecurrentPPO model from: {model_path}")
    model = RecurrentPPO.load(model_path)

    # Create single environment with viewer
    env = Go2GymSingle(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        lstm_states = None
        episode_start = True

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            episode_start = done

        print(f"Episode {ep+1}/{args.episodes} — total_reward: {total_reward:.3f}")

    env.env.close()
    print("✅ Evaluation finished.")


if __name__ == "__main__":
    main()

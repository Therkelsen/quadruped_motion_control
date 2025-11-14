import argparse
import os
import csv
import pickle
import numpy as np
import genesis as gs

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import test

from src.Gymwrapper import Go2GymSingle


def load_env(log_dir):
    """Load cfgs.pkl and construct evaluation environment."""
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfgs_path):
        raise FileNotFoundError(f"cfgs.pkl not found in {log_dir}")

    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(open(cfgs_path, "rb"))

    env = Go2GymSingle(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,    # disable viewer for batch eval
    )
    return env, env_cfg, obs_cfg, reward_cfg, command_cfg


def evaluate_model(model, env, episodes=10, vec_env=False):
    """Evaluate a model for a given number of episodes."""
    results = []

    # Helper to get the actual environment inside VecEnv
    def get_unwrapped_env(env):
        if hasattr(env, "envs"):  # VecEnv
            return env.envs[0]
        else:
            return env

    unwrapped_env = get_unwrapped_env(env)

    for ep in range(episodes):
        obs = env.reset() if vec_env else env.reset()[0]

        done = False
        ep_reward = 0.0
        ep_length = 0
        test_reward = 0.0

        while not done:
            # TD3/SAC deterministic evaluation
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)

            # Reward components from the actual environment
            track_vel_rew = unwrapped_env.env._reward_tracking_lin_vel().item()
            track_ang_rew = unwrapped_env.env._reward_tracking_ang_vel().item()
            lin_z_rew = unwrapped_env.env._reward_lin_vel_z().item()
            action_rate_rew = unwrapped_env.env._reward_action_rate().item()
            similar_default_rew = unwrapped_env.env._reward_similar_to_default().item()
            base_height_rew = unwrapped_env.env._reward_base_height().item()

            test_reward += (
                track_vel_rew
                + track_ang_rew
                + lin_z_rew
                + action_rate_rew
                + similar_default_rew
                + base_height_rew
            )

            # Print reward breakdown
            print(f"Tracking Linear Vel Reward: {track_vel_rew:.3f}")
            print(f"Tracking Angular Vel Reward: {track_ang_rew:.3f}")
            print(f"Linear Z Velocity Reward: {lin_z_rew:.3f}")
            print(f"Action Rate Reward: {action_rate_rew:.3f}")
            print(f"Similar to Default Pose Reward: {similar_default_rew:.3f}")
            print(f"Base Height Reward: {base_height_rew:.3f}")


            # Unpack step outputs depending on environment type
            if vec_env:
                obs, reward, dones, infos = step_out
                reward = reward[0]
                done = dones[0]
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated

            ep_reward += reward
            ep_length += 1

        # Episode summary
        print(f"Episode {ep+1}/{episodes} — Reward: {ep_reward:.3f}, Length: {ep_length}")
        print(f"Test_reward {test_reward:.3f}")

        results.append((ep_reward, ep_length))

    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--exp_root", type=str, default="logs")
    parser.add_argument("--output", type=str, default="evaluation_results.csv")
    args = parser.parse_args()

    gs.init()

    algorithms = [
        ("td3", TD3),
        ("ppo", PPO),
        ("sac", SAC),
    ]

    all_results = []

    for algo_name, algo_class in algorithms:
        print(f"\n=== Evaluating {algo_name.upper()} ===")

        log_dir = os.path.join(args.exp_root, algo_name.upper())
        model_path = os.path.join(log_dir, f"{algo_name}.zip")

        print(model_path)
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model for {algo_name} not found, skipping.")
            continue

        # Load environment configs
        env, env_cfg, obs_cfg, reward_cfg, command_cfg = load_env(log_dir)
        
        print("Episode length check:")
        print(env.env.max_episode_length)
        print(env_cfg["episode_length_s"])
        
        # TD3 requires VecNormalize + DummyVecEnv
        if algo_name == "td3":
            def make_env():
                return Go2GymSingle(
                    env_cfg=env_cfg,
                    obs_cfg=obs_cfg,
                    reward_cfg=reward_cfg,
                    command_cfg=command_cfg,
                    show_viewer=False,
                )

            env = DummyVecEnv([make_env])

            vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")
            if os.path.exists(vecnorm_path):
                env = VecNormalize.load(vecnorm_path, env)
                env.training = False
                env.norm_reward = False
                vec_env = True
            else:
                vec_env = True
        else:
            vec_env = False

        # Load model
        model = algo_class.load(model_path, env if vec_env else None)

        # Evaluate model
        results = evaluate_model(model, env, args.episodes, vec_env)

        # Store results with model type
        for i, (reward, length) in enumerate(results):
            all_results.append([algo_name.upper(), i + 1, reward, length])

        print(f"Finished {algo_name.upper()} evaluation.")

        env.close()

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "episode", "reward", "length"])
        writer.writerows(all_results)

    print(f"\n✅ Evaluation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()

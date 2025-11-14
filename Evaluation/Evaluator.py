import argparse
import os
import csv
import pickle
import numpy as np
import genesis as gs

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
        show_viewer=False,
    )
    return env, env_cfg, obs_cfg, reward_cfg, command_cfg


def evaluate_model(model, env, episodes=10, vec_env=False):
    """Evaluate a model for a given number of episodes and log cumulative reward components."""
    results = []

    def get_unwrapped_env(env):
        if hasattr(env, "envs"):  # VecEnv
            return env.envs[0]
        else:
            return env

    unwrapped_env = get_unwrapped_env(env)

    reward_scales = {
        "tracking_lin_vel": 5.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -1.0,
        "base_height": -5.0,
        "action_rate": -0.005,
        "similar_to_default": -0.1,
    }

    for ep in range(episodes):
        obs = env.reset() if vec_env else env.reset()[0]

        done = False
        ep_length = 0
        ep_rewards = {k: 0.0 for k in reward_scales}  # cumulative per component

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)

            # Compute and scale rewards
            ep_rewards["tracking_lin_vel"] += unwrapped_env.env._reward_tracking_lin_vel().item() * reward_scales["tracking_lin_vel"]
            ep_rewards["tracking_ang_vel"] += unwrapped_env.env._reward_tracking_ang_vel().item() * reward_scales["tracking_ang_vel"]
            ep_rewards["lin_vel_z"] += unwrapped_env.env._reward_lin_vel_z().item() * reward_scales["lin_vel_z"]
            ep_rewards["base_height"] += unwrapped_env.env._reward_base_height().item() * reward_scales["base_height"]
            ep_rewards["action_rate"] += unwrapped_env.env._reward_action_rate().item() * reward_scales["action_rate"]
            ep_rewards["similar_to_default"] += unwrapped_env.env._reward_similar_to_default().item() * reward_scales["similar_to_default"]

            # Unpack step outputs depending on env type
            if vec_env:
                obs, reward, dones, infos = step_out
                done = dones[0]
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated

            ep_length += 1

        ep_total = sum(ep_rewards.values())
        print(f"Episode {ep+1}/{episodes} — Total Reward: {ep_total:.3f}, Length: {ep_length}")
        results.append({
            "total": ep_total,
            "length": ep_length,
            **ep_rewards
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--exp_root", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    gs.init()

    os.makedirs(args.output_dir, exist_ok=True)

    algorithms = [
        ("td3", TD3),
        ("ppo", PPO),
        ("sac", SAC),
    ]

    for algo_name, algo_class in algorithms:
        print(f"\n=== Evaluating {algo_name.upper()} ===")

        log_dir = os.path.join(args.exp_root, algo_name.upper())
        model_path = os.path.join(log_dir, f"{algo_name}.zip")

        if not os.path.exists(model_path):
            print(f"⚠️ Model for {algo_name} not found, skipping.")
            continue

        # Load environment configs
        env, env_cfg, obs_cfg, reward_cfg, command_cfg = load_env(log_dir)

        # TD3 requires VecNormalize + DummyVecEnv
        vec_env = False
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

        # Load model
        model = algo_class.load(model_path, env if vec_env else None)

        # Evaluate model
        results = evaluate_model(model, env, args.episodes, vec_env)

        # Write results to CSV per model
        csv_path = os.path.join(args.output_dir, f"{algo_name.upper()}_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "total_reward", "length",
                "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z",
                "base_height", "action_rate", "similar_to_default"
            ])
            for i, r in enumerate(results):
                writer.writerow([
                    i + 1,
                    r["total"],
                    r["length"],
                    r["tracking_lin_vel"],
                    r["tracking_ang_vel"],
                    r["lin_vel_z"],
                    r["base_height"],
                    r["action_rate"],
                    r["similar_to_default"]
                ])

        print(f"Finished {algo_name.upper()} evaluation. Results saved to {csv_path}")
        env.close()


if __name__ == "__main__":
    main()

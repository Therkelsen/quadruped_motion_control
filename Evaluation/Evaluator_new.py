import argparse
import os
import csv
import pickle
import numpy as np
import genesis as gs

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.Gymwrapper import Go2GymSingle


def load_cfgs(log_dir):
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfgs_path):
        raise FileNotFoundError(f"cfgs.pkl not found in {log_dir}")
    return pickle.load(open(cfgs_path, "rb"))


def build_env_for_algo(log_dir, algo_name, show_viewer=False):
    """Construct evaluation env for a given algorithm.

    If a VecNormalize pickle is present, create a DummyVecEnv and load
    VecNormalize so observations/rewards are normalized the same way as
    during training. For TD3 we also prefer a vec env since training
    commonly used a VecNormalize wrapper.
    """
    env_cfg, obs_cfg, reward_cfg, command_cfg, _ = load_cfgs(log_dir)

    vecnorm_path = os.path.join(log_dir, "vecnormalize.pkl")
    use_vec = False
    if algo_name.lower() == "td3":
        use_vec = True
    if os.path.exists(vecnorm_path):
        use_vec = True

    if use_vec:
        def make_env():
            return Go2GymSingle(
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=show_viewer,
            )

        env = DummyVecEnv([make_env])
        if os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize from {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
        return env, True, env_cfg, obs_cfg, reward_cfg, command_cfg

    # single env (non-vectorized)
    env = Go2GymSingle(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    return env, False, env_cfg, obs_cfg, reward_cfg, command_cfg


def evaluate_model(model, env, episodes=10, vec_env=False):
    """Evaluate a model for a number of episodes and return detailed results.

    The function will try to call the env's internal component reward helpers
    (if present) and scale them the same way other tools in this repo do.
    """
    results = []

    def get_unwrapped_env(e):
        # try to reach the underlying single env where helper methods live
        if hasattr(e, "envs") and len(getattr(e, "envs", [])) > 0:
            candidate = e.envs[0]
        elif hasattr(e, "env"):
            candidate = e.env
        else:
            candidate = e

        # unwrap one more level if necessary
        if hasattr(candidate, "env"):
            return candidate.env
        return candidate

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
        # reset depending on env type
        if vec_env:
            obs = env.reset()
        else:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) >= 1 else reset_out

        done = False
        ep_length = 0
        ep_rewards = {k: 0.0 for k in reward_scales}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)

            # component rewards may not always be accessible; guard with try/except
            try:
                ep_rewards["tracking_lin_vel"] += unwrapped_env._reward_tracking_lin_vel().item() * reward_scales["tracking_lin_vel"]
                ep_rewards["tracking_ang_vel"] += unwrapped_env._reward_tracking_ang_vel().item() * reward_scales["tracking_ang_vel"]
                ep_rewards["lin_vel_z"] += unwrapped_env._reward_lin_vel_z().item() * reward_scales["lin_vel_z"]
                ep_rewards["base_height"] += unwrapped_env._reward_base_height().item() * reward_scales["base_height"]
                ep_rewards["action_rate"] += unwrapped_env._reward_action_rate().item() * reward_scales["action_rate"]
                ep_rewards["similar_to_default"] += unwrapped_env._reward_similar_to_default().item() * reward_scales["similar_to_default"]
            except Exception:
                # env doesn't expose component helpers — skip
                pass

            # unpack step outputs
            if vec_env:
                obs, reward, dones, infos = step_out
                done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated

            ep_length += 1

        ep_total = sum(ep_rewards.values())
        print(f"Episode {ep+1}/{episodes} — Total Reward: {ep_total:.3f}, Length: {ep_length}")
        results.append({
            "total": ep_total,
            "length": ep_length,
            **ep_rewards,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--exp_root", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--show_viewer", action="store_true")
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
            print(f"⚠️ Model for {algo_name} not found at {model_path}, skipping.")
            continue

        try:
            # Build evaluation environment consistently per-algorithm
            env, vec_env, env_cfg, obs_cfg, reward_cfg, command_cfg = build_env_for_algo(log_dir, algo_name, show_viewer=args.show_viewer)

            # Always pass the env to load so the model has the correct env attached
            print(f"Loading model from: {model_path} (vec_env={vec_env})")
            model = algo_class.load(model_path, env=env)

            # Evaluate
            results = evaluate_model(model, env, args.episodes, vec_env)

            # Save CSV
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
                        r.get("total", 0.0),
                        r.get("length", 0),
                        r.get("tracking_lin_vel", 0.0),
                        r.get("tracking_ang_vel", 0.0),
                        r.get("lin_vel_z", 0.0),
                        r.get("base_height", 0.0),
                        r.get("action_rate", 0.0),
                        r.get("similar_to_default", 0.0),
                    ])

            print(f"Finished {algo_name.upper()} evaluation. Results saved to {csv_path}")
        except Exception as e:
            print(f"Error during evaluation of {algo_name}: {e}")
        finally:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
import os
import argparse
import pickle
import shutil

import genesis as gs
from stable_baselines3 import SAC

from src.Gymwrapper import GenesisVecEnv
from src.Configs import get_cfgs, get_train_cfg

# SAC Training Script for Go2 Quadruped
# Key changes to make SAC work:
# 1. Use same reward scales as PPO (they are validated)
# 2. Increase num_envs to 256 for better diversity
# 3. Reduce learning_starts to 5k (less random action damage)
# 4. Update every step (train_freq=1) for faster convergence
# 5. Lower tau to 0.01 for stability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="SAC")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    # ---------------- Environment + training setup ----------------
    # Use more environments for better exploration and signal diversity
    num_envs = 256
    replay_buffer_size = 1_000_000
    sac_batch_size = 256
    learning_starts = 5_000  # Reduced: let it start learning sooner
    tau = 0.01  # Slightly higher for more stable updates
    train_freq = 1  # Update every step for faster learning
    gradient_steps = 1

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ---------------- Load configs ----------------
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, max_iterations=100)

    # ✅ Use SAME reward scales as PPO for fair comparison
    # These scales have been validated to work for this task
    # NOTE: Go2Env multiplies all reward scales by dt=0.02, so we must
    # provide scales that when multiplied by 0.02 give the desired values
    # E.g., to get effective scale of 1.0, we set it to 1.0/0.02 = 50.0
    reward_cfg["reward_scales"] = {
        "tracking_lin_vel": 1.0 / 0.02,        # 50.0 → effective 1.0
        "tracking_ang_vel": 0.2 / 0.02,        # 10.0 → effective 0.2
        "lin_vel_z": -1.0 / 0.02,              # -50.0 → effective -1.0
        "base_height": -50.0 / 0.02,           # -2500.0 → effective -50.0
        "action_rate": -0.005 / 0.02,          # -0.25 → effective -0.005
        "similar_to_default": -0.1 / 0.02,    # -5.0 → effective -0.1
    }

    # Save configs
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # ---------------- Create Genesis-based VecEnv ----------------
    vec_env = GenesisVecEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # ---------------- Define SAC parameters ----------------
    ent_coef = train_cfg["algorithm"].get("entropy_coef", "auto")
    learning_rate = train_cfg["algorithm"].get("learning_rate", 3e-4)
    gamma = train_cfg["algorithm"].get("gamma", 0.99)
    target_entropy = -0.5 * vec_env.action_space.shape[0]

    # ============ DIAGNOSTIC PRINTS FOR SAC SPACE NORMALIZATION ============
    print("\n" + "="*80)
    print("🔍 SAC SPACE NORMALIZATION DIAGNOSTICS")
    print("="*80)
    
    # Action space diagnostics
    print("\n📍 ACTION SPACE:")
    print(f"  Shape: {vec_env.action_space.shape}")
    print(f"  Low:   {vec_env.action_space.low[:3]}... (first 3 of {len(vec_env.action_space.low)})")
    print(f"  High:  {vec_env.action_space.high[:3]}... (first 3 of {len(vec_env.action_space.high)})")
    print(f"  Type:  {type(vec_env.action_space)}")
    
    # Check if action space is [-1, 1] (SAC expects normalized actions)
    action_low_ok = (vec_env.action_space.low == -1.0).all()
    action_high_ok = (vec_env.action_space.high == 1.0).all()
    print(f"  ✅ Actions in [-1, 1]? {action_low_ok and action_high_ok}")
    if not (action_low_ok and action_high_ok):
        print("     ⚠️  WARNING: Action space NOT normalized to [-1, 1]!")
    
    # Observation space diagnostics
    print("\n📍 OBSERVATION SPACE:")
    print(f"  Shape: {vec_env.observation_space.shape}")
    print(f"  Low:   {vec_env.observation_space.low[:5]}... (first 5 of {len(vec_env.observation_space.low)})")
    print(f"  High:  {vec_env.observation_space.high[:5]}... (first 5 of {len(vec_env.observation_space.high)})")
    print(f"  Type:  {type(vec_env.observation_space)}")
    
    # Sample initial observations to check scale
    print("\n📍 INITIAL OBSERVATION STATISTICS (from reset):")
    initial_obs = vec_env.reset()
    print(f"  Shape: {initial_obs.shape}")
    print(f"  Mean: {initial_obs.mean(axis=0)[:5]}... (first 5 dims)")
    print(f"  Std:  {initial_obs.std(axis=0)[:5]}... (first 5 dims)")
    print(f"  Min:  {initial_obs.min():.4f}")
    print(f"  Max:  {initial_obs.max():.4f}")
    print("  ⚠️  Check: Observations should have reasonable scale (~[-1, 1] is ideal)")
    
    # Reward scale diagnostics
    print("\n📍 REWARD SCALES:")
    for reward_name, scale in reward_cfg["reward_scales"].items():
        print(f"  {reward_name:25s}: {scale:8.4f}")
    
    # SAC hyperparameter diagnostics
    print("\n📍 SAC HYPERPARAMETERS:")
    print(f"  num_envs:         {num_envs}")
    print(f"  batch_size:       {sac_batch_size}")
    print(f"  learning_rate:    {learning_rate}")
    print(f"  gamma:            {gamma}")
    print(f"  tau (soft update): {tau}")
    print(f"  learning_starts:  {learning_starts} steps (random exploration phase)")
    print(f"  train_freq:       {train_freq} (updates per step)")
    print(f"  gradient_steps:   {gradient_steps} (gradient steps per update)")
    print(f"  ent_coef:         {ent_coef}")
    print(f"  target_entropy:   {target_entropy:.4f} (should be ~-action_dim/2 = {-0.5 * vec_env.action_space.shape[0]:.4f})")
    print(f"  replay_buffer_size: {replay_buffer_size}")
    
    print("\n" + "="*80 + "\n")
    # ========================================================================

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        buffer_size=replay_buffer_size,
        batch_size=sac_batch_size,
        learning_starts=learning_starts,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
    )

    # ---------------- Train ----------------
    model.learn(total_timesteps=args.total_timesteps)

    # ---------------- Save model ----------------
    model.save(os.path.join(log_dir, "sac"))
    print(f"✅ Model saved at {log_dir}/sac.zip")

    vec_env.close()


if __name__ == "__main__":
    main()

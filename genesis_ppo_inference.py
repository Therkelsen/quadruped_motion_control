#!/usr/bin/env python3
"""
Inference runner for a PPO model trained on the Genesis Go2 environment.

Usage:
    python run_inference.py

Adjust MODEL_PATH, NUM_EPISODES, and RENDER as needed.
"""

import time
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from genesis_ppo import Go2GenesisEnv  # your fixed env file

# ----------------------
# Configuration
# ----------------------
MODEL_PATH = "./models/ppo_go2_genesis_900_steps.zip"  # path to your saved SB3 model
NUM_EPISODES = 1
MAX_STEPS_PER_EPISODE = 100
RENDER = True          # set to True to open Genesis viewer
SLEEP_PER_STEP = 0 #1.0 / 240.0  # viewer timestep; 0 for fastest execution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Load model
# ----------------------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH, device=DEVICE)
print("Model loaded. Device:", DEVICE)

# ----------------------
# Create env
# ----------------------
print(f"Creating Genesis environment (render={RENDER})...")
env = Go2GenesisEnv(render=RENDER, device=DEVICE)

def reset_env():
    """Handle gymnasium vs gym reset return."""
    ret = env.reset()
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0]
    return ret

# ----------------------
# Run episodes
# ----------------------
episode_rewards = []
try:
    for ep in range(NUM_EPISODES):
        obs = reset_env()
        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < MAX_STEPS_PER_EPISODE:
            # Use deterministic actions for inference
            action, _ = model.predict(obs, deterministic=True)

            # Step the environment
            step_ret = env.step(action)

            # Handle gymnasium vs gym API
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_ret

            ep_reward += float(reward)
            step += 1

            # Slow down for visualization
            if RENDER and SLEEP_PER_STEP > 0:
                time.sleep(SLEEP_PER_STEP)

        episode_rewards.append(ep_reward)
        print(f"[Episode {ep+1}/{NUM_EPISODES}] reward={ep_reward:.3f} steps={step}")

except KeyboardInterrupt:
    print("Interrupted by user. Closing environment...")

finally:
    env.close()
    if episode_rewards:
        print("Episodes completed:", len(episode_rewards))
        print("Reward stats: mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}".format(
            np.mean(episode_rewards),
            np.std(episode_rewards),
            np.min(episode_rewards),
            np.max(episode_rewards)
        ))
    else:
        print("No episodes finished.")

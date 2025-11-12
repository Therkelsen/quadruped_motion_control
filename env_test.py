import torch, genesis as gs, numpy as np
from src.Gymwrapper import Go2GymSingle
from src.Configs import get_cfgs

# --- Initialize Genesis ---
gs.init()

# --- Load configs ---
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

# --- 🔧 Increase actuation strength for visibility ---
env_cfg["action_scale"] = 0.5       # was 0.25 → stronger joint movement
env_cfg["kp"] = 80.0                # stiffer PD control
env_cfg["kd"] = 1.0
env_cfg["simulate_action_latency"] = False  # remove delay for clarity

# (optional) print current control parameters
print(f"Using action_scale={env_cfg['action_scale']}, kp={env_cfg['kp']}, kd={env_cfg['kd']}")

# --- Create environment ---
env = Go2GymSingle(env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True)

# --- Check initial joint configuration ---
print("Default DOF positions:", env.env.default_dof_pos.cpu().numpy())

# --- Random action test loop ---
obs, _ = env.reset()
for t in range(200):
    # sample random actions uniformly in the valid action space
    a = env.action_space.sample()

    obs, rew, term, trunc, info = env.step(a)

    if term:
        print(f"Episode terminated early at step {t}")
        obs, _ = env.reset()

env.close()
print("random action test done ✅")

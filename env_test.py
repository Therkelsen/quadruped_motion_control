# quick_random_test.py
import torch, genesis as gs, numpy as np
from src.Gymwrapper import Go2GymSingle
from src.Configs import get_cfgs
gs.init()
env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
env = Go2GymSingle(env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True)

obs, _ = env.reset()
for t in range(200):
    # sample random actions in action_space
    a = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(a)
    if term: 
        obs, _ = env.reset()
env.close()
print("random action test done")

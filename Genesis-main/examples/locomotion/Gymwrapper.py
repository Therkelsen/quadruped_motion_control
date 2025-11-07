import gymnasium as gym
import numpy as np
import torch
from go2_env import Go2Env

class Go2GymWrapper(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_envs=1, env_cfg=None, obs_cfg=None, reward_cfg=None, command_cfg=None, render_mode=None):
        super().__init__()
        self.env = Go2Env(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg)
        self.num_envs = num_envs

        # ✅ handle default configs in case None is passed
        if obs_cfg is None:
            obs_cfg = {"num_obs": 45}
        if env_cfg is None:
            env_cfg = {"num_actions": 12}

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_cfg["num_obs"],),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env_cfg["num_actions"],),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset()
        obs_np = obs[0].detach().cpu().numpy()  # ✅ ensure tensor → numpy safely
        info = {}
        return obs_np, info

    def step(self, action):
        # ✅ handle both numpy and torch tensors
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        obs, reward, done, info = self.env.step(action)

        obs_np = obs[0].detach().cpu().numpy()
        reward_f = float(reward.item())
        terminated = bool(done.item())
        truncated = False  # unless you implement time limits

        return obs_np, reward_f, terminated, truncated, {}

    def render(self):
        # optional — depends on your sim
        pass

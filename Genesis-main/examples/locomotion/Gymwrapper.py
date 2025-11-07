# gymwrapper.py
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from typing import Dict, Any, List, Tuple

from go2_env import Go2Env
import genesis as gs


class GenesisVecEnv(VecEnv):
    def __init__(self, num_envs: int, env_cfg: dict, obs_cfg: dict, reward_cfg: dict,
                 command_cfg: dict, show_viewer: bool = False):

        obs_dim = obs_cfg["num_obs"]
        action_dim = env_cfg["num_actions"]

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        self.env = Go2Env(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                          reward_cfg=reward_cfg, command_cfg=command_cfg,
                          show_viewer=show_viewer)

        self.device = gs.device if hasattr(gs, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._obs = None
        self._rewards = None
        self._dones = None
        self._infos = None
        self.actions_pending = None
        self.closed = False

    def reset(self) -> np.ndarray:
        """Reset the vectorized environment and return observations.

        SB3 expects reset() from a VecEnv to return a plain ndarray of
        observations (shape [num_envs, obs_dim]). Previously this returned
        a tuple (obs, info) which made Stable-Baselines3 receive a tuple and
        raise "Unrecognized type of observation <class 'tuple'>".
        """
        obs, _ = self.env.reset()
        # ensure obs is a tensor (Go2Env may return a tensor or a tuple-wrapped tensor)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_np = obs.detach().cpu().numpy()
        # force 2D shape [num_envs, obs_dim]
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]
        self._obs = obs_np
        return obs_np


    def step_async(self, actions: np.ndarray) -> None:
        self.actions_pending = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if self.actions_pending is None:
            raise RuntimeError("step_wait called without step_async")

        act_t = torch.tensor(self.actions_pending, dtype=torch.float32, device=self.device)
        obs_t, reward_t, done_t, infos = self.env.step(act_t)

        # unwrap obs if tuple
        if isinstance(obs_t, tuple):
            obs_t = obs_t[0]

        obs_np = obs_t.detach().cpu().numpy()
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]

        rewards = reward_t.detach().cpu().numpy() if isinstance(reward_t, torch.Tensor) else np.array(reward_t, dtype=np.float32)
        dones = done_t.detach().cpu().numpy().astype(bool) if isinstance(done_t, torch.Tensor) else np.array(done_t, dtype=bool)

        per_env_infos: List[Dict[str, Any]] = []
        for i in range(self.num_envs):
            ie: Dict[str, Any] = {}
            if dones[i]:
                # SB3 expects episode info under 'episode' key
                ep_info: Dict[str, float] = {"r": float(rewards[i])}  # total reward for this episode
                # merge any additional episode info from infos
                if isinstance(infos, dict) and "episode" in infos:
                    for k, v in infos["episode"].items():
                        try:
                            if isinstance(v, torch.Tensor):
                                val = v[i].item() if v.numel() == self.num_envs else v.item()
                            elif isinstance(v, np.ndarray):
                                val = v[i].item() if v.size == self.num_envs else v.item()
                            elif isinstance(v, (list, tuple)):
                                val = v[i]
                            else:
                                val = v
                        except Exception:
                            val = float(v) if hasattr(v, "__float__") else v
                        ep_info[k] = val
                ie["episode"] = ep_info
            per_env_infos.append(ie)

        self._obs, self._rewards, self._dones, self._infos = obs_np, rewards, dones, per_env_infos
        self.actions_pending = None

        return obs_np, rewards, dones, per_env_infos




    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        return None

    def close(self):
        if not self.closed:
            try:
                self.env.close()
            except Exception:
                pass
            self.closed = True

    def seed(self, seed: int = None):
        return
    
    def env_is_wrapped(self, wrapper_class) -> bool:
        """Required by VecEnv: return whether the wrapped env(s) are wrapped by a given wrapper class."""
        return False


    # SB3 VecEnv abstract methods
    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:
            indices = range(self.num_envs)
        results = []
        for i in indices:
            results.append(getattr(self.env, method_name)(*args, **kwargs))
        return results

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.env, attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for _ in indices:
            setattr(self.env, attr_name, value)


class Go2GymSingle(gym.Env):
    def __init__(self, env_cfg: dict, obs_cfg: dict, reward_cfg: dict, command_cfg: dict, show_viewer: bool = True):
        super().__init__()
        self.env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                          reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=show_viewer)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_cfg["num_obs"],), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(env_cfg["num_actions"],), dtype=np.float32)
        self.device = gs.device if hasattr(gs, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        obs, _ = self.env.reset()
        return obs[0].detach().cpu().numpy(), {}

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs, reward, done, info = self.env.step(action)
        obs_np = obs[0].detach().cpu().numpy()
        reward_f = float(reward[0].item()) if isinstance(reward, torch.Tensor) else float(reward[0])
        terminated = bool(done[0].item()) if isinstance(done, torch.Tensor) else bool(done[0])
        truncated = False
        return obs_np, reward_f, terminated, truncated, info

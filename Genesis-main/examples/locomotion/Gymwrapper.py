# gymwrapper.py
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from typing import Dict, Any, Sequence, Tuple, List

# import your Go2Env and genesis device
from go2_env import Go2Env
import genesis as gs


class GenesisVecEnv(VecEnv):
    """
    A vectorized environment that wraps Go2Env's internal vectorized simulator.
    This exposes SB3's VecEnv API while letting Go2Env simulate all envs inside one GPU process.
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        show_viewer: bool = False,
    ):
        # observation and action spaces (sb3 expects gym spaces)
        # we create spaces from provided cfgs
        obs_dim = obs_cfg["num_obs"]
        action_dim = env_cfg["num_actions"]

        # build gym spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        super().__init__(num_envs, self.observation_space, self.action_space)

        # instantiate the underlying Go2Env with vectorized simulation (all envs inside this process/GPU)
        # set show_viewer if you want a viewer (useful for eval; not recommended for large num_envs)
        self.env = Go2Env(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )

        # device to place tensors on (genesis exposes device)
        self.device = gs.device if hasattr(gs, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # track last observations etc.
        self._obs = None
        self._dones = None
        self._rewards = None
        self._infos = None

        # required by VecEnv: a step_async / step_wait implementation
        self.actions_pending = None
        self.closed = False

    # --- VecEnv API ---
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments.
        Returns:
            obs: np.ndarray of shape (num_envs, obs_dim)
            infos: dict (gymnasium style)
        """
        obs, _ = self.env.reset()
        # obs is expected to be a torch tensor shape [num_envs, obs_dim]
        obs_np = obs.detach().cpu().numpy()
        self._obs = obs_np
        # SB3 expects (obs, infos) for gymnasium compatibility. We'll return infos as empty dict.
        return obs_np, {}

    def step_async(self, actions: np.ndarray) -> None:
        """
        Stores actions to be executed by step_wait. actions shape: (num_envs, action_dim)
        """
        self.actions_pending = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Execute the previously stored actions (from step_async) and return results.
        Returns:
            obs (np.ndarray): shape (num_envs, obs_dim)
            rewards (np.ndarray): shape (num_envs,)
            dones (np.ndarray): shape (num_envs,) (bools)
            infos (list[dict]): list of info dicts, length = num_envs
        """
        if self.actions_pending is None:
            raise RuntimeError("step_wait called without step_async")

        # convert numpy actions to torch tensor and move to genesis device
        act_np = np.asarray(self.actions_pending, dtype=np.float32)
        act_t = torch.from_numpy(act_np).to(self.device)

        # ensure correct shape [num_envs, action_dim]
        # call env.step with batched actions (Go2Env expects a torch tensor)
        obs_t, reward_t, done_t, infos = self.env.step(act_t)

        # convert outputs to numpy
        obs_np = obs_t.detach().cpu().numpy()
        # rewards/dones may be tensors of shape [num_envs, 1] or [num_envs]
        if isinstance(reward_t, torch.Tensor):
            rewards = reward_t.detach().cpu().squeeze(-1).numpy().astype(np.float32)
        else:
            rewards = np.asarray(reward_t, dtype=np.float32).squeeze(-1)

        if isinstance(done_t, torch.Tensor):
            dones = done_t.detach().cpu().squeeze(-1).numpy().astype(bool)
        else:
            dones = np.asarray(done_t, dtype=bool).squeeze(-1)

        # infos: rsl-rl returns infos as dict with keys, but SB3 expects a list of dicts per env
        # If infos is a dict with arrays, convert to list of per-env dicts. If it's already a list, keep it.
        per_env_infos: List[Dict[str, Any]] = []
        if isinstance(infos, dict) and "observations" in infos:
            # convert to minimal per-env list (user may extend)
            # produce one dict per env, optionally include 'episode' from infos if present
            num_envs = obs_np.shape[0]
            for i in range(num_envs):
                ie = {}
                # episode logs if present
                if "episode" in infos:
                    try:
                        ep = infos["episode"][i]
                        ie["episode"] = ep
                    except Exception:
                        pass
                per_env_infos.append(ie)
        elif isinstance(infos, list):
            per_env_infos = infos
        else:
            # default: empty dict per env
            per_env_infos = [{} for _ in range(obs_np.shape[0])]

        # store last values
        self._obs = obs_np
        self._rewards = rewards
        self._dones = dones
        self._infos = per_env_infos

        # clear pending actions
        self.actions_pending = None

        return obs_np, rewards, dones, per_env_infos

    # convenience: vectorized step that combines async+wait (SB3 may call this)
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode: str = "human"):
        # If Go2Env supports rendering (show_viewer), it's handled internally.
        # We don't need to do anything here.
        return None

    def close(self):
        if not self.closed:
            try:
                self.env.close()
            except Exception:
                pass
            self.closed = True

    def seed(self, seed: int = None):
        # optional: implement if Go2Env supports seeding
        return

    def env_is_wrapped(self, wrapper_class) -> bool:
        # Not used normally, return False
        return False


# --- small single-env wrapper for evaluation when you want show_viewer=True ---
class Go2GymSingle(gym.Env):
    def __init__(self, env_cfg: dict, obs_cfg: dict, reward_cfg: dict, command_cfg: dict, show_viewer: bool = True):
        super().__init__()
        self.env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=show_viewer)
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
        reward_f = float(reward.item())
        terminated = bool(done.item())
        truncated = False
        return obs_np, reward_f, terminated, truncated, info

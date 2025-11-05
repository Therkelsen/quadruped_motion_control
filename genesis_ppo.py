import time
import torch
import gymnasium as gym
import numpy as np
import genesis as gs

from genesis.utils.geom import quat_to_rotvec
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# --------------------------------------------------------------------
# Initialize Genesis
# --------------------------------------------------------------------
try:
    gs.init(backend=gs.gpu)
except Exception:
    print("⚠️ GPU backend not available — falling back to CPU.")
    gs.init(backend=gs.cpu)


class Go2GenesisEnv(gym.Env):
    """Quadruped Go2-like environment using Genesis v0.9+."""
    def __init__(self, render=False, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.steps_taken = 0
        self.max_steps = 10000000
        self.target = np.array([2.0, 2.0], dtype=np.float32)
        self.last_action = None
        self.dt = 1 / 240.0
        self.dt = 0.02  # control frequency on real robot is 50hz
        num_envs = 1

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.5, 0.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, show_world_frame=False),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=render,
        )


        # -----------------------------
        # Scene
        # -----------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt, 
                substeps=10,
                gravity=[0, 0, -9.81]
            ),
            viewer_options=gs.options.ViewerOptions(max_FPS=int(1 / self.dt)),
            vis_options=gs.options.VisOptions(show_world_frame=False),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            show_viewer=render
        )

        # -----------------------------
        # Plane
        # -----------------------------
        self.scene.add_entity(gs.morphs.URDF(file="/home/peter/uni/ai_project/quadruped_motion_control/objects/plane/plane.urdf", fixed=True))

        # -----------------------------
        # Robot
        # -----------------------------
        # Do NOT pass pos/quat here — avoids creating an offset (fixed) base joint.
        # Base pose is set in reset().
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/peter/uni/ai_project/quadruped_motion_control/objects/go2/urdf/go2.urdf",
                fixed=False,
            )
        )
        
        self.other_ball = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.06,
                pos=torch.tensor([1.0, 1.0, 3.0], device=self.device, dtype=torch.float32).cpu().numpy(),
                quat=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).cpu().numpy(),
                fixed=False,
            )
        )
        
        self.ball = self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/peter/uni/ai_project/quadruped_motion_control/objects/ball/ball.urdf",
                pos=torch.tensor([0.0, 3.0, 3.0], device=self.device, dtype=torch.float32).cpu().numpy(),
                quat=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).cpu().numpy(),
                fixed=False,
            )
        )

        self.scene.build(n_envs=1)

        # -----------------------------
        # Joints
        # -----------------------------
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.joint_ids = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names]
        self.num_joints = len(self.joint_ids)

        # -----------------------------
        # Spaces
        # -----------------------------
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + (2 * self.num_joints) + 4 + 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # quick check: print base_joint info if present
        try:
            bj = self.robot.get_joint("base_joint")
            print("BASE JOINT:", getattr(bj, "type", getattr(bj, "joint_type", bj)))
        except Exception:
            print("no explicit base_joint found or cannot query it")

        # --- Debug probe: inspect robot / URDF dynamics ---
        try:
            print("DEBUG: robot pos / quat:", getattr(self.robot, "get_pos", lambda: None)(), getattr(self.robot, "get_quat", lambda: None)())
            print("DEBUG: robot vel / ang:", getattr(self.robot, "get_vel", lambda: None)(), getattr(self.robot, "get_ang", lambda: None)())
            print("DEBUG: robot dir():", [k for k in dir(self.robot) if not k.startswith("_")][:40])
            # Print joint dof indices to ensure joints are actuated/free
            try:
                print("DEBUG: joint dof indices:", [(name, self.robot.get_joint(name).dof_idx_local) for name in self.joint_names])
            except Exception as e:
                print("DEBUG: could not list joints:", e)

            # Try giving the robot a small downward velocity to test dynamics
            if hasattr(self.robot, "set_vel"):
                print("DEBUG: applying small downward vel to robot...")
                try:
                    self.robot.set_vel([0.0, 0.0, -0.5])
                    for _ in range(10):
                        self.scene.step()
                    print("DEBUG: robot pos after vel probe:", getattr(self.robot, "get_pos", lambda: None)())
                except Exception as e:
                    print("DEBUG: set_vel probe failed:", e)
            else:
                print("DEBUG: robot has no set_vel API exposed.")
        except Exception as e:
            print("DEBUG: robot inspection failed:", e)
        # --- end debug ---

    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0

        # Safe base start (start slightly above ground to avoid initial penetration)
        safe_z = 0.6
        try:
            self.robot.set_pos([0.0, 0.0, safe_z])
            self.robot.set_quat([0.0, 0.0, 0.0, 1.0])
        except Exception:
            pass

        # Zero base and joint velocities if API available
        try:
            self.robot.zero_all_dofs_velocity()
        except Exception:
            try:
                self.robot.set_dofs_velocity(torch.zeros(self.num_joints, device=self.device), self.joint_ids)
            except Exception:
                pass

        # Initialize a reasonable standing joint pose (12 DOFs: 3 per leg)
        standing_pose = torch.tensor([0.0, 0.8, -1.5] * 4, device=self.device, dtype=torch.float32)
        try:
            self.robot.set_dofs_position(standing_pose, self.joint_ids)
            self.robot.set_dofs_velocity(torch.zeros_like(standing_pose), self.joint_ids)
        except Exception:
            # fallback: control position directly per DOF if API differs
            try:
                self.robot.control_dofs_position(standing_pose, self.joint_ids)
            except Exception:
                pass

        # Let physics settle so no NaNs and stable contact states
        for _ in range(60):
            self.scene.step()

        # Randomize target and reset last action
        self.target = np.random.uniform(low=-5.0, high=5.0, size=2).astype(np.float32)
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)

        return self._get_obs(), {}

    # ==========================================================
    def step(self, action):
        self.steps_taken += 1
        action = np.clip(action, -1.0, 1.0)
        action_t = torch.tensor(action, device=self.device, dtype=torch.float32)

        # Small delta to joint positions
        max_delta = np.deg2rad(15)
        current_positions = torch.tensor(self.robot.get_dofs_position(self.joint_ids), device=self.device, dtype=torch.float32)
        target_positions = torch.clamp(current_positions + action_t * max_delta, -1.0, 1.0)

        # Control DOFs
        self.robot.control_dofs_position(target_positions, self.joint_ids)

        # Step simulation
        self.scene.step()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self.steps_taken >= self.max_steps
        truncated = False

        self.last_action = action.copy()
        return obs, float(reward), terminated, truncated, {}

    # ==========================================================
    def _compute_reward(self, action):
        base_pos = torch.tensor(self.robot.get_pos(), device=self.device, dtype=torch.float32).cpu().numpy().flatten()
        dist = np.linalg.norm(base_pos[:2] - self.target)
        reward = -dist - 0.01 * np.sum(np.square(action))
        return reward

    # ==========================================================
    def _get_obs(self):
        def _to_np(x):
            try:
                return np.asarray(x, dtype=np.float32).ravel()
            except Exception:
                return np.array([], dtype=np.float32)

        # collect components (as numpy arrays)
        base_pos = _to_np(self.robot.get_pos())
        base_quat = _to_np(self.robot.get_quat())
        base_lin_vel = _to_np(self.robot.get_vel())
        base_ang_vel = _to_np(self.robot.get_ang())
        # convert quaternion -> rotvec (guarded)
        try:
            euler = quat_to_rotvec(base_quat).astype(np.float32)
        except Exception:
            euler = np.zeros(3, dtype=np.float32)

        joint_pos = _to_np(self.robot.get_dofs_position(self.joint_ids))
        joint_vel = _to_np(self.robot.get_dofs_velocity(self.joint_ids))

        contact_vec = np.zeros(4, dtype=np.float32)
        for i, link_name in enumerate(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]):
            try:
                link_id = self.robot.get_link(link_name).idx
                contact_vec[i] = 1.0 if self.robot.get_contacts(link_id) else 0.0
            except Exception:
                contact_vec[i] = 0.0

        components = {
            "euler": euler,
            "base_pos": base_pos,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "contact_vec": contact_vec,
            "target": np.array(self.target, dtype=np.float32),
        }

        # detect NaNs and replace with zeros (report once)
        any_nan = False
        nan_sources = []
        for k, v in components.items():
            if np.isnan(v).any():
                any_nan = True
                nan_sources.append(k)
                components[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        if any_nan and not getattr(self, "_nan_reported", False):
            print("DEBUG: NaNs detected in observation components:", nan_sources)
            self._nan_reported = True

        obs = np.concatenate([
            components["euler"],
            components["base_pos"],
            components["base_lin_vel"],
            components["base_ang_vel"],
            components["joint_pos"],
            components["joint_vel"],
            components["contact_vec"],
            components["target"],
        ]).astype(np.float32)

        return obs

    # ==========================================================
    def close(self):
        self.scene.destroy()


# ============================================================
# Training loop
# ============================================================
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def make_env():
        env = Go2GenesisEnv(render=True, device=device)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path="./models/",
        name_prefix="ppo_go2_genesis"
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", device=device)
    total_timesteps = 1_000_000

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save("./models/ppo_go2_genesis_latest")

    env.close()

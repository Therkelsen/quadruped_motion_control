import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
import torch.optim as optim

from ppo_ball_train import BallEnv  # Reuse the environment from ppo.py

# ----------------------------
# Neural network dynamics model
# ----------------------------
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # model predicts the change (delta) in state: next_state - state
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # predict delta state
        )

    def forward(self, state, action):
        # both inputs expected as (batch, dim)
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


# ----------------------------
# PETS agent skeleton
# ----------------------------
class PETSAgent:
    def __init__(self, env, ensemble_size=5):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.ensemble_size = ensemble_size
        self.models = [DynamicsModel(self.state_dim, self.action_dim) for _ in range(ensemble_size)]
        self.optimizers = [optim.Adam(m.parameters(), lr=1e-3) for m in self.models]

        # Dataset for model-based learning (kept as numpy until processed)
        self.dataset = {"states": [], "actions": [], "next_states": []}

        # normalization stats (set after collecting data)
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.delta_mean = None
        self.delta_std = None

    def collect_random_data(self, steps=1000):
        state, _ = self.env.reset()
        for i in range(steps):
            print(f"Collecting data step {i+1}/{steps}", end="\r")
            action = self.env.action_space.sample()
            next_state, _, terminated, truncated, _ = self.env.step(action)
            self.dataset["states"].append(np.array(state, dtype=np.float32))
            self.dataset["actions"].append(np.array(action, dtype=np.float32))
            self.dataset["next_states"].append(np.array(next_state, dtype=np.float32))
            state = next_state
            if terminated or truncated:
                state, _ = self.env.reset()

        # convert to numpy arrays
        for k in list(self.dataset.keys()):
            self.dataset[k] = np.array(self.dataset[k], dtype=np.float32)

        # compute deltas and normalization stats
        deltas = self.dataset["next_states"] - self.dataset["states"]
        self.state_mean = self.dataset["states"].mean(axis=0)
        self.state_std = self.dataset["states"].std(axis=0) + 1e-8
        self.action_mean = self.dataset["actions"].mean(axis=0)
        self.action_std = self.dataset["actions"].std(axis=0) + 1e-8
        self.delta_mean = deltas.mean(axis=0)
        self.delta_std = deltas.std(axis=0) + 1e-8

        # create normalized torch tensors for training
        states_n = (self.dataset["states"] - self.state_mean) / self.state_std
        actions_n = (self.dataset["actions"] - self.action_mean) / self.action_std
        deltas_n = (deltas - self.delta_mean) / self.delta_std

        self.dataset = {
            "states": torch.tensor(states_n, dtype=torch.float32),
            "actions": torch.tensor(actions_n, dtype=torch.float32),
            "deltas": torch.tensor(deltas_n, dtype=torch.float32)
        }

    def train_dynamics(self, epochs=50, batch_size=64):
        n = self.dataset["states"].size(0)
        loss_fn = nn.MSELoss()

        # create train/validation split (10% val)
        idxs = torch.randperm(n)
        val_n = max(1, int(0.1 * n))
        val_idx = idxs[:val_n]
        train_idx = idxs[val_n:]

        for idx_model, (m, opt) in enumerate(zip(self.models, self.optimizers)):
            m.train()
            for ep in range(epochs):
                # bootstrap sample indices for ensemble diversity (train split)
                perm = train_idx[torch.randint(0, train_idx.size(0), (train_idx.size(0),))]
                epoch_losses = []
                for i in range(0, perm.size(0), batch_size):
                    idx = perm[i:i+batch_size]
                    state_b = self.dataset["states"][idx]
                    action_b = self.dataset["actions"][idx]
                    target_b = self.dataset["deltas"][idx]
                    pred = m(state_b, action_b)
                    loss = loss_fn(pred, target_b)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_losses.append(loss.item())

                # compute validation loss for this epoch (on val split)
                with torch.no_grad():
                    state_v = self.dataset["states"][val_idx]
                    action_v = self.dataset["actions"][val_idx]
                    target_v = self.dataset["deltas"][val_idx]
                    pred_v = m(state_v, action_v)
                    val_loss = loss_fn(pred_v, target_v).item()

                avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
                print(f"Model {idx_model} Epoch {ep+1}/{epochs} train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f}")

        # After training all models, print evaluation metrics on full dataset
        self.evaluate_models()

    def _normalize(self, state_np, action_np=None):
        # return normalized torch tensors (batch dim = 1)
        s = (state_np - self.state_mean) / self.state_std
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        if action_np is None:
            return s_t
        a = (action_np - self.action_mean) / self.action_std
        a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        return s_t, a_t

    def plan_action(self, state, horizon=40, num_samples=40):
        state_np = np.array(state, dtype=np.float32)
        # sample all action sequences at once: (num_samples, horizon, action_dim)
        action_seqs = np.random.uniform(self.env.action_space.low,
                                        self.env.action_space.high,
                                        size=(num_samples, horizon, self.action_dim)).astype(np.float32)

        # initialize batch states: (num_samples, state_dim)
        sim_states = np.tile(state_np, (num_samples, 1))
        total_rewards = np.zeros(num_samples, dtype=np.float32)

        # normalize once
        state_mean = self.state_mean
        state_std = self.state_std
        action_mean = self.action_mean
        action_std = self.action_std
        delta_mean = self.delta_mean
        delta_std = self.delta_std

        for t in range(horizon):
            actions_t = action_seqs[:, t, :]  # (num_samples, action_dim)

            # normalize
            s_t = torch.tensor((sim_states - state_mean)/state_std, dtype=torch.float32)
            a_t = torch.tensor((actions_t - action_mean)/action_std, dtype=torch.float32)

            # ensemble predictions (vectorized)
            ensemble_preds = []
            with torch.no_grad():
                for m in self.models:
                    ensemble_preds.append(m(s_t, a_t))  # (num_samples, state_dim)
                delta_pred_norm = torch.stack(ensemble_preds).mean(dim=0)
            
            # unnormalize
            delta_pred = delta_pred_norm.numpy() * delta_std + delta_mean
            sim_states += delta_pred

            # reward
            pos = sim_states[:, 0:2]
            dist = np.linalg.norm(pos - self.env.target, axis=1)
            total_rewards += -dist - 0.01 * np.sum(np.square(actions_t), axis=1)

        # pick best sequence
        best_idx = np.argmax(total_rewards)
        return action_seqs[best_idx, 0, :]


    def evaluate_models(self, batch_size=1024):
        """Evaluate each ensemble member and the ensemble mean on the full collected dataset.
        Reports per-dimension RMSE for predicted state deltas (in original scale) and overall RMSE.
        """
        # dataset tensors are normalized; targets are normalized deltas
        states = self.dataset["states"]
        actions = self.dataset["actions"]
        targets_norm = self.dataset["deltas"]

        device = next(self.models[0].parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        targets_norm = targets_norm.to(device)

        delta_mean_t = torch.tensor(self.delta_mean, dtype=torch.float32, device=device)
        delta_std_t = torch.tensor(self.delta_std, dtype=torch.float32, device=device)

        all_model_rmse = []
        all_model_rmse_per_dim = []

        preds_per_model = []

        with torch.no_grad():
            # evaluate each model in batches
            for m_idx, m in enumerate(self.models):
                m.eval()
                preds_norm_list = []
                for i in range(0, states.size(0), batch_size):
                    s_b = states[i:i+batch_size]
                    a_b = actions[i:i+batch_size]
                    pred_norm = m(s_b, a_b)  # normalized delta
                    preds_norm_list.append(pred_norm)
                preds_norm = torch.cat(preds_norm_list, dim=0)

                # unnormalize to original delta scale
                preds_unnorm = preds_norm * delta_std_t + delta_mean_t
                targets_unnorm = targets_norm * delta_std_t + delta_mean_t

                mse_per_dim = ((preds_unnorm - targets_unnorm) ** 2).mean(dim=0).cpu().numpy()
                rmse_per_dim = np.sqrt(mse_per_dim)
                rmse_overall = float(np.sqrt(mse_per_dim.mean()))

                all_model_rmse.append(rmse_overall)
                all_model_rmse_per_dim.append(rmse_per_dim)
                preds_per_model.append(preds_unnorm.cpu().numpy())

                print(f"Model {m_idx} delta RMSE per-dim: {rmse_per_dim.round(6)} overall RMSE: {rmse_overall:.6f}")

            # ensemble mean prediction
            preds_stack = np.stack(preds_per_model, axis=0)  # (ensemble, N, state_dim)
            ensemble_mean_preds = preds_stack.mean(axis=0)
            targets_unnorm_np = (targets_norm * delta_std_t + delta_mean_t).cpu().numpy()

            mse_ens_per_dim = np.mean((ensemble_mean_preds - targets_unnorm_np) ** 2, axis=0)
            rmse_ens_per_dim = np.sqrt(mse_ens_per_dim)
            rmse_ens_overall = float(np.sqrt(mse_ens_per_dim.mean()))
            print(f"Ensemble mean delta RMSE per-dim: {rmse_ens_per_dim.round(6)} overall RMSE: {rmse_ens_overall:.6f}")

        return {
            "models_rmse": all_model_rmse,
            "models_rmse_per_dim": all_model_rmse_per_dim,
            "ensemble_rmse": rmse_ens_overall,
            "ensemble_rmse_per_dim": rmse_ens_per_dim
        }
# ...existing code...

# ----------------------------
# Training loop
# ----------------------------
if __name__ == "__main__":
    # Data collection / training: headless
    env_collect = BallEnv(human_friendly=False)
    agent = PETSAgent(env_collect)

    print("Collecting random data (headless)...")
    agent.collect_random_data(steps=10_000)

    print("Training dynamics model (headless)...")
    agent.train_dynamics(epochs=200)

    # Close headless env before opening GUI to avoid multiple PyBullet connections
    try:
        env_collect.close()
    except Exception:
        pass

    # MPC control: GUI
    env_gui = BallEnv(human_friendly=True)
    env_gui.use_terminate = True

    # Reuse trained agent but point it to the GUI env for planning/execute
    agent.env = env_gui

    print("Running MPC control (GUI)...")
    state, _ = env_gui.reset()
    while True:
        state, _ = env_gui.reset()
        for t in range(500):
            action = agent.plan_action(state)
            state, reward, terminated, truncated, _ = env_gui.step(action)
            #print(f"Timestep {t}, reward {reward:.3f}")
            if terminated or truncated:
                break

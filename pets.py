import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
import torch.optim as optim

from ppo import MyCustomEnv  # Reuse the environment from ppo.py

# ----------------------------
# Neural network dynamics model
# ----------------------------
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # predict next state
        )

    def forward(self, state, action):
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

        # Dataset for model-based learning
        self.dataset = {"states": [], "actions": [], "next_states": []}

    def collect_random_data(self, steps=1000):
        state, _ = self.env.reset()
        for _ in range(steps):
            action = self.env.action_space.sample()
            next_state, _, _, _, _ = self.env.step(action)
            self.dataset["states"].append(state)
            self.dataset["actions"].append(action)
            self.dataset["next_states"].append(next_state)
            state = next_state

        # Convert to torch tensors
        self.dataset = {k: torch.tensor(np.array(v), dtype=torch.float32) for k,v in self.dataset.items()}

    def train_dynamics(self, epochs=50, batch_size=64):
        for m, opt in zip(self.models, self.optimizers):
            for _ in range(epochs):
                perm = torch.randperm(self.dataset["states"].size(0))
                for i in range(0, len(perm), batch_size):
                    idx = perm[i:i+batch_size]
                    state = self.dataset["states"][idx]
                    action = self.dataset["actions"][idx]
                    target = self.dataset["next_states"][idx]
                    pred = m(state, action)
                    loss = nn.MSELoss()(pred, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def plan_action(self, state, horizon=5, num_samples=100):
        """Random shooting MPC"""
        best_reward = -float("inf")
        best_action = None
        for _ in range(num_samples):
            actions = np.random.uniform(self.env.action_space.low,
                                        self.env.action_space.high,
                                        size=(horizon, self.action_dim))
            sim_state = torch.tensor(state, dtype=torch.float32)
            total_reward = 0
            for a in actions:
                # Predict next state with first model (simplified)
                sim_state = self.models[0](sim_state.unsqueeze(0), torch.tensor(a, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                
                total_reward += self.env.compute_reward()
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = actions[0]
        return best_action


# ----------------------------
# Training loop
# ----------------------------
if __name__ == "__main__":
    env = MyCustomEnv(human_friendly=True)
    agent = PETSAgent(env)

    # Step 1: collect random data
    print("Collecting random data...")
    agent.collect_random_data(steps=2000)

    # Step 2: train dynamics model
    print("Training dynamics model...")
    agent.train_dynamics(epochs=100)

    # Step 3: MPC loop
    print("Running MPC control...")    
    state, _ = env.reset()
    for t in range(500):
        action = agent.plan_action(state)
        state, reward, done, _, _ = env.step(action)
        print(f"Timestep {t}, reward {reward:.3f}")
        if done:
            state, _ = env.reset()

    env.close()

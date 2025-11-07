import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ppo import MyCustomEnv

# ----------------------------
# Dynamics Neural Network
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
# TensorBoard + Checkpoint Callback
# ----------------------------
class ModelCheckpointCallback:
    def __init__(self, save_dir="./models", save_freq=5_000):
        self.save_dir = save_dir
        self.save_freq = save_freq
        os.makedirs(save_dir, exist_ok=True)
        self.step_counter = 0

    def maybe_save(self, models):
        self.step_counter += 1
        if self.step_counter % self.save_freq == 0:
            for i, model in enumerate(models):
                torch.save(model.state_dict(), f"{self.save_dir}/dynamics_{i}.pth")
            print(f"[Checkpoint] Saved ensemble at step {self.step_counter}")


# ----------------------------
# PETS Trainer with Logging
# ----------------------------
class PETSTrainer:
    def __init__(self, env, ensemble_size=5, lr=1e-3, log_dir="./tensorboard"):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.ensemble_size = ensemble_size

        # Ensemble of dynamics models
        self.models = [DynamicsModel(self.state_dim, self.action_dim) for _ in range(ensemble_size)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]

        # Dataset
        self.dataset = {"states": [], "actions": [], "next_states": []}

        # TensorBoard writers
        self.writers = [
            SummaryWriter(log_dir=os.path.join(log_dir, f"model_{i}"))
            for i in range(ensemble_size)
        ]
        self.global_steps = [0] * ensemble_size

    def collect_random_data(self, steps=2000):
        state, _ = self.env.reset()
        for _ in range(steps):
            action = self.env.action_space.sample()
            next_state, _, done, _, _ = self.env.step(action)
            self.dataset["states"].append(state)
            self.dataset["actions"].append(action)
            self.dataset["next_states"].append(next_state)
            state = next_state
            if done:
                state, _ = self.env.reset()

        self.dataset = {k: torch.tensor(np.array(v), dtype=torch.float32) for k,v in self.dataset.items()}
        print(f"Collected {steps} samples for model training.")

    def train_forever(self, batch_size=64, checkpoint_freq=5000):
        """Train indefinitely, with TensorBoard logging and checkpointing."""
        callback = ModelCheckpointCallback(save_dir="./models", save_freq=checkpoint_freq)

        while True:
            for model_idx, (m, opt, writer) in enumerate(zip(self.models, self.optimizers, self.writers)):
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

                    # Logging
                    self.global_steps[model_idx] += 1
                    step = self.global_steps[model_idx]
                    writer.add_scalar("Loss/train", loss.item(), step)

                    # Periodic printing
                    if step % 1000 == 0:
                        print(f"[Model {model_idx}] Step {step}, Loss = {loss.item():.6f}")

                    # Checkpointing
                    callback.maybe_save(self.models)


if __name__ == "__main__":
    env = MyCustomEnv(human_friendly=True)
    trainer = PETSTrainer(env, ensemble_size=5)

    print("Collecting initial random data...")
    trainer.collect_random_data(steps=500_000)

    print("Starting indefinite training with TensorBoard + checkpoints...")
    trainer.train_forever(batch_size=128, checkpoint_freq=100_000)

# mpc_inference.py
import torch
import numpy as np
from ppo import MyCustomEnv
from pets import DynamicsModel

class PETSInference:
    def __init__(self, env, model_folder="models", ensemble_size=5):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.ensemble_size = ensemble_size

        self.models = [DynamicsModel(self.state_dim, self.action_dim) for _ in range(ensemble_size)]
        for i, m in enumerate(self.models):
            m.load_state_dict(torch.load(f"{model_folder}/dynamics_{i}.pth"))
            m.eval()
        print(f"Loaded {ensemble_size} models from {model_folder}/")

    def plan_action(self, state, horizon=5, num_samples=100):
        best_reward = -float("inf")
        best_action = None

        for _ in range(num_samples):
            actions = np.random.uniform(
                self.env.action_space.low,
                self.env.action_space.high,
                size=(horizon, self.action_dim)
            )

            sim_state = torch.tensor(state, dtype=torch.float32)
            total_reward = 0
            for a in actions:
                with torch.no_grad():
                    preds = [m(sim_state.unsqueeze(0), torch.tensor(a, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                             for m in self.models]
                # average prediction
                sim_state = torch.stack(preds).mean(0)
                total_reward += self.env._compute_reward(a)

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = actions[0]
        return best_action

if __name__ == "__main__":
    env = MyCustomEnv(human_friendly=True)
    agent = PETSInference(env)

    state, _ = env.reset()
    for t in range(500):
        action = agent.plan_action(state)
        state, reward, done, _, _ = env.step(action)
        print(f"Timestep {t}, reward {reward:.3f}")
        if done:
            state, _ = env.reset()

    env.close()

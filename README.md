# Quadruped Motion Control in Genesis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Genesis](https://img.shields.io/badge/Genesis-Simulation-orange)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-RL-brightgreen)

## Overview
This repository contains a complete reinforcement learning framework for training the [Unitree Go2](https://shop.unitree.com/products/unitree-go2?utm_term=go2+robot&utm_campaign=&utm_source=adwords&utm_medium=ppc&hsa_acc=8764137937&hsa_cam=16924829192&hsa_grp=138654326834&hsa_ad=593212817493&hsa_src=g&hsa_tgt=kwd-2176275031814&hsa_kw=go2+robot&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=16924829192&gbraid=0AAAAABa3bGsfX-DPqOys4fCeoD-sBksu_&gclid=Cj0KCQiApfjKBhC0ARIsAMiR_Itwe4hooTzzFutfFINgSNbOupPNVzfqUSMuDfgowjwJ4IbIN0flRKIaAnJeEALw_wcB) quadruped robot to achieve stable, efficient, and agile locomotion. Built on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulation environment and integrated with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), this project implements, evaluates, and statistically compares three state-of-the-art continuous control algorithms:
- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)
- **TD3** (Twin Delayed DDPG)

---

## System Architecture
- **Sim-to-Real Prepared Environment**: The core `Go2Env` features built-in **Domain Randomization**, injecting noise into joint positions, base coordinates, and quaternions upon episode resets to enforce robust policy learning.
- **Custom Gym Wrappers**: Implements tailored wrappers (`Go2GymSingle`, `GenesisVecEnv`) that bridge Genesis's PyTorch tensors with SB3's numpy-based Gym interface for high-throughput vectorized training.
- **Centralized Configuration**: `Configs.py` handles all environment parameters, URDF mapping, reward scaling, observation shaping, and hyperparameter tuning in one place.
- **Granular Reward Engineering**: Component-based reward tracking including linear/angular velocity tracking, Z-velocity penalties, base height maintenance, and action rate smoothing.
- **Statistical Analysis Suite**: Automated data pipelines utilizing `pandas`, `seaborn`, and `statsmodels` for distribution plotting, alongside a dedicated `manova.py` script for rigorous multivariate statistical validation (MANOVA, ANOVA, and Tukey HSD).

---

## Setup & Installation

**1. Clone the repository & initialize submodules**
```bash
git clone https://github.com/Therkelsen/quadruped_motion_control.git
cd quadruped_motion_control
git submodule update --init --recursive
```

**2. Configure the Python environment**
```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Environment Testing
Verify that the Genesis environment, rigid body dynamics, and URDF configurations load correctly before training:
```bash
python3 -m src.env_test
```

### 2. Training
Hyperparameters and reward scales can be tuned directly in `src/Configs.py`. 
Train a policy using PPO, SAC, or TD3. Checkpoints and tensorboard logs are saved automatically in the `/logs` directory.
```bash
# Train PPO (Default: 10M timesteps)
python3 -m Training.PPO_train --exp_name ppo_experiment --total_timesteps 10000000

# Train SAC or TD3
python3 -m Training.SAC_train --exp_name sac_experiment
python3 -m Training.TD3_train --exp_name td3_experiment
```
*Monitor training in real-time:* `tensorboard --logdir logs`

### 3. Evaluation & Visual Inference
Watch the trained policy control the robot in the Genesis GUI:
```bash
python3 -m Evaluation.PPO_eval --exp_name ppo_experiment --episodes 5
```

### 4. Batched Evaluation & Statistical Analysis
To benchmark all trained models (PPO, SAC, TD3) over multiple episodes and extract granular reward CSVs:
```bash
python3 -m Evaluation.Evaluator --episodes 30 --exp_root logs --output_dir data_analysis/data
```

**Visual Data Analysis:** Generate distribution grids (Boxplots, QQ-plots, Scatter matrices) saved to `data_analysis/figures/`.
```bash
python3 -m data_analysis.data_analysis --data data_analysis/data --exclude episode,length,action_rate
```

**Rigorous Hypothesis Testing:** Run MANOVA, one-way ANOVAs, and Tukey HSD post-hoc tests across your evaluation data to statistically compare algorithm performance:
```bash
python3 -m data_analysis.manova --data data_analysis/data --exclude episode,length,action_rate
```

---
## Resources
- [Genesis Embodied AI Simulator](https://github.com/Genesis-Embodied-AI/Genesis)
- [Unitree Go2 Specifications](https://shop.unitree.com/)
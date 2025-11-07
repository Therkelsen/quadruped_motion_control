# Quadruped Motion Control

The aim of this project is to develop locomotion controllers that
allow robots to walk with both efficiency and agility using 3 approaches to reinforcement learning.
Specifically, model-free learning with Proximal Policy
Optimization (PPO), model-based learning with Probabilistic
Ensembles and Trajectory Sampling (PETS), and imitation
learning through DeepMimic. Each is evaluated on its ability to
generate stable, efficient, and natural motion for the simulation of [Unitree A1](https://www.unitree.com/a1) in PyBullet.


## Setup

Clone repo:
```bash
git clone https://github.com/Therkelsen/quadruped_motion_control.git && cd quadruped_motion_control
```

Initialize submodule:
```bash
git submodule update --init --recursive
```

Setup virtual environment:
```bash
python3 -m venv env
```

Source it:
```bash
source env/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run programme

Run the training scripts for doing reinforcement learning (default total_timesteps is 10_000_000):

```bash
python3 -m Training.PPO_train.py --exp_name experiment_name --total_timesteps total_timesteps
```

Open a terminal and activate tensorboard logging:
```bash
tensorboard --logdir logs
```

For running inference:
```bash
python3 -m Evaluation.PPO_eval.py --exp_name experiment_name --episodes num_episodes
```

## Resources:

[Genesis Github](https://github.com/Genesis-Embodied-AI/Genesis?tab=readme-ov-file)
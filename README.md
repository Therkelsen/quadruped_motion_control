# Quadruped Motion Control

The aim of this project is to develop locomotion controllers that
allow robots to walk with both efficiency and agility using 3 approaches to reinforcement learning.
Specifically, model-free learning with Proximal Policy
Optimization (PPO), model-based learning with Probabilistic
Ensembles and Trajectory Sampling (PETS), and imitation
learning through DeepMimic. Each is evaluated on its ability to
generate stable, efficient, and natural motion for the simulation of [Unitree A1](https://www.unitree.com/a1) in PyBullet.


## Setup
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
Open a terminal and activate tensorboard logging:
```bash
tensorboard --logdir ./tensorboard
```

Run the programme ppo.py for doing ppo reinforcement learning:

```bash
ppo.py
```

## Resources:

[URDF File](https://github.com/unitreerobotics/unitree_ros/tree/master/robots)
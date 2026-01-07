# Quadruped Motion Control

The aim of this project is to develop locomotion controllers that
allow robots to walk with both efficiency and agility using 3 approaches to reinforcement learning.
Specifically the algorithms; Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and Twin Delayed DDPG (TD3). 
Each is evaluated on its ability to generate stable, efficient, and natural motion for the simulation of [Unitree Go2](https://shop.unitree.com/products/unitree-go2?utm_term=go2+robot&utm_campaign=&utm_source=adwords&utm_medium=ppc&hsa_acc=8764137937&hsa_cam=16924829192&hsa_grp=138654326834&hsa_ad=593212817493&hsa_src=g&hsa_tgt=kwd-2176275031814&hsa_kw=go2+robot&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=16924829192&gbraid=0AAAAABa3bGsfX-DPqOys4fCeoD-sBksu_&gclid=Cj0KCQiApfjKBhC0ARIsAMiR_Itwe4hooTzzFutfFINgSNbOupPNVzfqUSMuDfgowjwJ4IbIN0flRKIaAnJeEALw_wcB) in The Genesis environment.


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
python3 -m venv .env
```

Source it:
```bash
source .env/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run programme

Run the training scripts for doing reinforcement learning (default total_timesteps is 10_000_000):

```bash
python3 -m Training.PPO_train --exp_name experiment_name --total_timesteps total_timesteps
```

Open a terminal and activate tensorboard logging:
```bash
tensorboard --logdir logs
```

For running inference:
```bash
python3 -m Evaluation.PPO_eval --exp_name experiment_name --episodes num_episodes
```

The same commands apply for using TD3 and SAC, just replace fx. PPO_train with TD3_train and PPO_eval with TD3_eval and so on.

The Evaluator Tool under Evaluation can be used to evaluate several models in one go and print csv data for the episodes for later use by the scripts in the data_analysis folder.

Data analysis:
```bash
python3 data_analysis/data_analysis.py --data data_analysis/data --exclude episode,length,action_rate
```

## Resources:

[Genesis Github](https://github.com/Genesis-Embodied-AI/Genesis?tab=readme-ov-file)
# Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning (ICML 2025)
This repository considers the implementation of the paper "Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning" which has been accepted to ICML 2025, and is available at [https://arxiv.org/abs/2502.02834](https://arxiv.org/abs/2502.02844). This repository is based on the PyMARL implementation of MARL baselines([https://github.com/katerakelly/oyster](https://github.com/oxwhirl/pymarl)).

## Abstract
Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenarios. To address this limitation, we propose the Wolfpack Adversarial Attack framework, inspired by wolf hunting strategies, which targets an initial agent and its assisting agents to disrupt cooperation. Additionally, we introduce the Wolfpack-Adversarial Learning for MARL (WALL) framework, which trains robust MARL policies to defend against the proposed Wolfpack attack by fostering system-wide collaboration. Experimental results underscore the devastating impact of the Wolfpack attack and the significant robustness improvements achieved by WALL.

<img width="319" height="205" alt="thumbnail_ICML" src="https://github.com/user-attachments/assets/9e5649bb-ce83-4ef8-949f-fb15a22eb957" />

## Installation instructions

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment.

## Run an WALL experiment

QMIX
```shell
python3 src/main.py --config=wall_qmix --env-config=sc2 with env_args.map_name=2s3z num_attack_test=8 num_attack_train=12 num_followup_agents=2 pretrain=True 
```
VDN
```shell
python3 src/main.py --config=wall_vdn --env-config=sc2 with env_args.map_name=2s3z num_attack_test=8 num_attack_train=12 num_followup_agents=2 pretrain=True 
```
QPLEX
```shell
python3 src/main.py --config=wall_qplex --env-config=sc2 with env_args.map_name=2s3z num_attack_test=8 num_attack_train=12 num_followup_agents=2 pretrain=True 
```

## Publication

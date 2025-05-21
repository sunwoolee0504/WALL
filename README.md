# Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning (ICML 2025)

## Installation instructions

## Abstract


Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

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

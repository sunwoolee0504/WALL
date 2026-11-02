# Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning (ICML 2025)
This repository considers the implementation of the paper "Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning" which has been accepted to ICML 2025, and is available at [https://arxiv.org/abs/2502.02844](https://arxiv.org/abs/2502.02844). This repository is based on the PyMARL implementation of MARL baselines([https://github.com/oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)).

## Abstract
Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenarios. To address this limitation, we propose the Wolfpack Adversarial Attack framework, inspired by wolf hunting strategies, which targets an initial agent and its assisting agents to disrupt cooperation. Additionally, we introduce the Wolfpack-Adversarial Learning for MARL (WALL) framework, which trains robust MARL policies to defend against the proposed Wolfpack attack by fostering system-wide collaboration. Experimental results underscore the devastating impact of the Wolfpack attack and the significant robustness improvements achieved by WALL.

<img width="319" height="205" alt="thumbnail_ICML" src="https://github.com/user-attachments/assets/9e5649bb-ce83-4ef8-949f-fb15a22eb957" />

## Installation instructions

Use Linux with Python 3.10 or 3.11 (including `venv`), Git, `curl`, and `unzip`.
The installer creates a new virtual environment and installs the dependencies in
`requirements.txt` with the selected PyTorch CPU/CUDA wheel.

```shell
git clone https://github.com/sunwoolee0504/WALL.git
cd WALL

WALL_PYTHON_BIN=python3.11 bash scripts/install_optimized_env.sh cu121 .venv-wall
source .venv-wall/bin/activate

bash scripts/install_sc2.sh "$PWD/3rdparty/StarCraftII"
export SC2PATH="$PWD/3rdparty/StarCraftII"
python scripts/check_environment.py
```

Use `python3.10` instead of `python3.11` if needed. The example selects the
CUDA 12.1 wheel; choose `cu118`, `cu124`, or `cpu` to suit your system. If
StarCraft II 4.10 and the SMAC maps are already installed, set `SC2PATH` to that
installation instead of running the SC2 installer. Activate the environment and
set `SC2PATH` again when opening a new shell.

## Run a WALL experiment

Run the following commands from the repository root with the environment active.
Each example trains on `2s3z` using the corresponding pretrained defender in
`pretrain_model/<algorithm>/2s3z/`.

**QMIX**

```shell
python src/main.py --config=wall_qmix --env-config=sc2 with \
  env_args.map_name=2s3z pretrain=True seed=1
```

**VDN**

```shell
python src/main.py --config=wall_vdn --env-config=sc2 with \
  env_args.map_name=2s3z pretrain=True seed=1
```

**QPLEX**

```shell
python src/main.py --config=wall_qplex --env-config=sc2 with \
  env_args.map_name=2s3z pretrain=True seed=1
```

Attack settings are loaded from `src/config/map_presets.yaml` for the selected
map. For `2s3z`, these are `num_attack_train=12`, `num_attack_test=8`, and
`num_followup_agents=2`; append any of these options to override the defaults.

The default training budget is 10,000 environment steps. Append `t_max=<steps>`
to set the budget for your experiment, or `use_cuda=False` to run on CPU.
Use `pretrain=False` to train from scratch. Logs and checkpoints are saved under
`results/` by default.

## Publication

```bibtex
@InProceedings{pmlr-v267-lee25h,
  title     = {Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning},
  author    = {Lee, Sunwoo and Hwang, Jaebak and Jo, Yonghyeon and Han, Seungyul},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = {33025--33056},
  year      = {2025},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v267/lee25h.html}
}
```

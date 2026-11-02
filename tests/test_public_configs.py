"""Public WALL configuration/registry consistency without SMAC or network access."""
from copy import deepcopy
import importlib
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from components.action_selectors import REGISTRY as action_registry
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as controller_registry
from envs import REGISTRY as env_registry
from learners import REGISTRY as learner_registry
from modules.agents import REGISTRY as agent_registry
from modules.step.models import Planning_Transformer, Qdifference_Transformer
from run import args_sanity_check
from runners import REGISTRY as runner_registry


PUBLIC_ALGORITHMS = {
    "natural_qmix": ("q_learner", "qmix"),
    "wall_qmix": ("q_learner", "qmix"),
    "wall_vdn": ("q_learner", "vdn"),
    "wall_qplex": ("dmaq_qatten_learner", "dmaq"),
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def test_public_files_and_registries_match_supported_algorithms():
    algorithms = {path.stem for path in (ROOT / "src/config/algs").glob("*.yaml")}
    environments = {path.stem for path in (ROOT / "src/config/envs").glob("*.yaml")}
    assert algorithms == set(PUBLIC_ALGORITHMS)
    assert environments == {"sc2"}
    assert set(learner_registry) == {"q_learner", "dmaq_qatten_learner"}
    assert set(runner_registry) == {"episode"}
    assert set(controller_registry) == {"basic_mac"}
    assert set(agent_registry) == {"rnn"}
    assert set(action_registry) == {"epsilon_greedy"}
    # Other tests inject toy environments; the published configuration resolves
    # to this production factory without launching a simulator.
    assert callable(env_registry["sc2"])


def test_all_public_python_modules_import_without_simulator_or_telemetry():
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT / "src").with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        if parts:
            importlib.import_module(".".join(parts))


@pytest.mark.parametrize("algorithm,expected", PUBLIC_ALGORITHMS.items())
def test_merged_public_config_constructs_supported_controller_predictors_and_learner(algorithm, expected):
    cfg = yaml.safe_load((ROOT / "src/config/default.yaml").read_text())
    for path in (ROOT / "src/config/envs/sc2.yaml", ROOT / ("src/config/algs/" + algorithm + ".yaml")):
        cfg = merge(cfg, yaml.safe_load(path.read_text()))
    cfg.update(use_cuda=False, use_amp=False)
    fixed = args_sanity_check(cfg, logging.getLogger(__name__))
    assert (fixed["learner"], fixed["mixer"]) == expected
    assert fixed["agent_output_type"] == "q"
    assert fixed["runner"] in runner_registry
    assert fixed["env"] in env_registry
    assert fixed["mac"] in controller_registry
    assert fixed["agent"] in agent_registry
    assert fixed["action_selector"] in action_registry
    args = SimpleNamespace(**fixed)
    args.n_agents, args.n_actions = 3, 9
    args.state_shape, args.obs_shape = 48, 30
    args.device = "cpu"
    scheme = {"state": {"vshape": args.state_shape},
              "obs": {"vshape": args.obs_shape, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long}}
    groups = {"agents": args.n_agents}
    batch = EpisodeBatch(scheme, groups, 1, 2,
                         preprocess={"actions": ("actions_onehot", [OneHot(args.n_actions)])})
    controller = controller_registry[args.mac](batch.scheme, groups, args)
    learner = learner_registry[args.learner](
        controller, batch.scheme, None, args,
        Qdifference_Transformer(args), Planning_Transformer(args))
    learner.to("cpu")
    assert next(learner.mac.parameters()).device.type == "cpu"
    assert all(not parameter.requires_grad for parameter in learner.target_mac.parameters())

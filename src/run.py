"""WALL experiment setup, reproducible evaluation and bounded replay staging."""
from contextlib import contextmanager
from copy import deepcopy
import datetime
import logging
import math
import os
from pathlib import Path
import pprint
import random
import time
from types import SimpleNamespace as SN

import numpy as np
import torch as th
import yaml

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.logging import Logger
from utils.timehelper import time_str

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Source of truth includes the author's correction to Table C.3 for 3m.
with (PROJECT_ROOT / "src/config/map_presets.yaml").open() as _preset_file:
    _map_config = yaml.safe_load(_preset_file)
WALL_MAP_PRESETS = {
    name: tuple(settings[key] for key in ("num_attack_train", "num_attack_test", "num_followup_agents"))
    for name, settings in _map_config.items()
}



def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    th.use_deterministic_algorithms(bool(deterministic))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        th.backends.cudnn.benchmark = False


@contextmanager
def isolated_rng(seed=None):
    """Evaluation never advances the training Python/NumPy/Torch RNG streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [th.cuda.current_device()] if th.cuda.is_initialized() else []
    try:
        with th.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                th.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def args_sanity_check(config, _log):
    config = deepcopy(config)
    if config.get("runner", "episode") != "episode" or config.get("batch_size_run", 1) != 1:
        raise ValueError("WALL currently supports runner=episode and batch_size_run=1; parallel runner has no WALL attack path")
    if config.get("use_cuda", False) and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA is unavailable; using CPU for models, replay and attacks")
    if config.get("use_amp", False) and not config.get("use_cuda", False):
        config["use_amp"] = False
        _log.warning("AMP is disabled on the CPU fallback")
    supported = {"env": ("sc2",), "mac": ("basic_mac",), "agent": ("rnn",),
                 "action_selector": ("epsilon_greedy",), "agent_output_type": ("q",),
                 "mixer": ("vdn", "qmix", "dmaq"),
                 "learner": ("q_learner", "dmaq_qatten_learner"),
                 "attention_backend": ("sdpa", "auto", "math"),
                 "init_attack_step_method": ("planner", "random")}
    for key, choices in supported.items():
        if key in config and config[key] not in choices:
            raise ValueError("{} must be one of {}".format(key, ", ".join(choices)))
    if config.get("learner") == "dmaq_qatten_learner" and config.get("mixer") != "dmaq":
        raise ValueError("dmaq_qatten_learner requires mixer=dmaq")
    map_name = config.get("env_args", {}).get("map_name", "3m")
    preset = WALL_MAP_PRESETS.get(map_name)
    for index, key in enumerate(("num_attack_train", "num_attack_test", "num_followup_agents")):
        if config.get(key) is None:
            if preset is None:
                raise ValueError("No WALL map preset for {!r}; set {} explicitly".format(map_name, key))
            config[key] = preset[index]
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 0:
            raise ValueError("{} must be a nonnegative integer".format(key))
    positive = ("batch_size", "buffer_size", "test_nepisode", "test_interval", "log_interval",
                "attack_period", "trans_input_len", "trans_embed_dim",
                "num_trans_head", "num_trans_layer", "planner_batch_size", "planner_origins_per_episode",
                "cpu_threads", "target_update_interval", "t_max", "runner_log_interval",
                "learner_log_interval", "save_model_interval", "rnn_hidden_dim", "trans_ff_multiplier",
                "epsilon_anneal_time")
    for key in positive:
        if key in config and (not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0):
            raise ValueError("{} must be a positive integer".format(key))
    for key in ("attack_duration", "planner_warmup_updates"):
        if key in config and (not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 0):
            raise ValueError("{} must be a nonnegative integer".format(key))
    if config.get("buffer_size", 32) < config.get("batch_size", 32):
        raise ValueError("buffer_size must be at least batch_size")
    if config.get("trans_embed_dim", 64) % config.get("num_trans_head", 1):
        raise ValueError("trans_embed_dim must be divisible by num_trans_head")
    span = config.get("planner_train_span")
    if span is None:
        span = max(config.get("trans_input_len", 20), config.get("attack_period", 20))
    if not isinstance(span, int) or isinstance(span, bool) or span < config.get("attack_period", 20):
        raise ValueError("planner_train_span must be a positive integer at least attack_period")
    config["planner_train_span"] = span
    for key in ("planner_lr", "qdiff_lr"):
        if config.get(key) is None:
            config[key] = config.get("lr", 0.0005)
    for key in ("temperature", "lr", "planner_lr", "qdiff_lr", "followup_q_lr", "grad_norm_clip", "optim_eps"):
        value = config.get(key, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(key))
    for key in ("world_model_aux_weight", "planner_state_loss_weight", "planner_obs_loss_weight",
                "planner_loss_weight", "qdifference_loss_weight"):
        value = config.get(key, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and nonnegative".format(key))
    if not 0 <= config.get("trans_dropout", 0.1) < 1:
        raise ValueError("trans_dropout must lie in [0, 1)")
    if not 0 <= config.get("gamma", 0.99) <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    if not 0 <= config.get("epsilon_finish", 0.05) <= config.get("epsilon_start", 1.0) <= 1:
        raise ValueError("epsilon probabilities must satisfy 0 <= finish <= start <= 1")
    if not 0 <= config.get("random_attack_probability", 0.1) <= 1:
        raise ValueError("random_attack_probability must lie in [0, 1]")
    if not config.get("test_greedy", True):
        raise ValueError("WALL evaluation uses greedy actions; test_greedy must be True")
    if config.get("followup_l2", False):
        config["followup_selection"] = "l2"
    if config.get("target_update_mode") == "soft":
        config["target_update_mode"] = "ema"
    if config.get("target_update_mode", "hard") not in ("hard", "ema"):
        raise ValueError("target_update_mode must be hard or ema (soft is an alias)")
    if not 0 < config.get("target_update_tau", 0.005) <= 1:
        raise ValueError("target_update_tau must lie in (0, 1]")
    if config.get("amp_dtype", "bfloat16") not in ("bfloat16", "float16"):
        raise ValueError("amp_dtype must be bfloat16 or float16")
    if config.get("followup_selection", "q_gradient") not in ("q_gradient", "l2"):
        raise ValueError("followup_selection must be q_gradient or l2")
    if not config.get("init_agent_random", True):
        raise ValueError("The paper-aligned state-only scheduler requires init_agent_random=True")
    config["test_nepisode"] = max(1, config.get("test_nepisode", 32))
    config.setdefault("seed", 1)
    config.setdefault("env_args", {})["seed"] = config["seed"]
    for key in ("train_attack_predictors", "evaluate_wolfpack"):
        if not isinstance(config.get(key, True), bool):
            raise ValueError("{} must be boolean".format(key))
    return config


def _project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _checkpoint_directory(path, load_step=0):
    path = _project_path(path)
    if not path.is_dir():
        raise FileNotFoundError("Checkpoint directory does not exist: {}".format(path))
    if (path / "learner.th").exists() or (path / "agent.th").exists():
        return path, int(path.name) if path.name.isdigit() else 0
    timesteps = [int(child.name) for child in path.iterdir() if child.is_dir() and child.name.isdigit()]
    if not timesteps:
        raise FileNotFoundError("No checkpoint or numeric timestep directories in {}".format(path))
    step = max(timesteps) if not load_step else min(timesteps, key=lambda value: abs(value - load_step))
    return path / str(step), step


def _load_local_checkpoint(path):
    # Local checkpoints intentionally contain NumPy/Python RNG and replay objects.
    return th.load(str(path), map_location="cpu", weights_only=False)


def run(_run, _config, _log):
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    config = args_sanity_check(_config, _log)
    args = SN(**config)
    args.device = "cuda" if args.use_cuda else "cpu"
    th.set_num_threads(args.cpu_threads)
    seed_everything(args.seed, args.deterministic)
    args.local_results_path = str(_project_path(args.local_results_path))
    args.unique_token = "{}__{}__seed{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"), args.seed)
    logger = Logger(_log)
    _log.info("Experiment Parameters:\n%s", pprint.pformat(vars(args), indent=2))
    try:
        if args.use_tensorboard:
            logger.setup_tb(Path(args.local_results_path) / "tb_logs" / args.unique_token)
        if args.use_wandb:
            directory = Path(args.local_results_path) / "wandb" / args.unique_token
            directory.mkdir(parents=True, exist_ok=True)
            logger.setup_wandb(args, directory)
        logger.setup_sacred(_run)
        run_sequential(args, logger)
    finally:
        logger.close()


def _evaluate(args, runner, seed):
    with isolated_rng(seed):
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True)
            if getattr(args, "evaluate_wolfpack", True):
                runner.run_wolfpack_attacker(test_mode=True)
        if args.save_replay:
            runner.save_replay()


def run_sequential(args, logger):
    # Optional SMAC and model dependencies are loaded only when an experiment runs.
    from learners import REGISTRY as learner_registry
    from runners import REGISTRY as runner_registry
    from controllers import REGISTRY as mac_registry
    from modules.step.models import Qdifference_Transformer, Planning_Transformer

    runner = runner_registry[args.runner](args=args, logger=logger)
    eval_runner = None
    try:
        env_info = runner.get_env_info()
        for key in ("n_agents", "n_actions", "state_shape", "obs_shape"):
            setattr(args, key, env_info[key])
        if args.num_followup_agents >= args.n_agents:
            raise ValueError("num_followup_agents must be smaller than n_agents")
        args.episode_limit = env_info["episode_limit"]
        args.env_max_timestep = max(args.env_max_timestep, env_info["episode_limit"] + args.attack_period + args.trans_input_len + 1)
        scheme = {
            "state": {"vshape": args.state_shape},
            "obs": {"vshape": args.obs_shape, "group": "agents"},
            "initial_agent": {"vshape": 1, "dtype": th.long},
            "followup_agents": {"vshape": args.num_followup_agents, "dtype": th.long},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "forced_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {"agents": args.n_agents}
        preprocess = {key: (key + "_onehot", [OneHot(out_dim=args.n_actions)])
                      for key in ("actions", "forced_actions")}
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)
        mac = mac_registry[args.mac](buffer.scheme, groups, args)
        qdifference_transformer, planning_transformer = Qdifference_Transformer(args), Planning_Transformer(args)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac,
                     qdifference_transformer=qdifference_transformer, planning_transformer=planning_transformer)
        learner = learner_registry[args.learner](mac, buffer.scheme, logger, args, qdifference_transformer, planning_transformer)
        learner.to(args.device)
        runner.setup_learner(learner)
        runner.setup_mac_for_attack(mac)
        runtime = {"episode": 0, "last_test_T": -args.test_interval - 1, "last_log_T": 0, "model_save_time": 0}
        if args.checkpoint_path:
            model_path, step = _checkpoint_directory(args.checkpoint_path, args.load_step)
            logger.console_logger.info("Loading checkpoint from %s", model_path)
            learner.load_models(str(model_path), resume=args.resume_checkpoint)
            if args.resume_checkpoint:
                runner.t_env = step
                runtime_path = model_path / "runtime_state.th"
                if runtime_path.exists():
                    loaded = _load_local_checkpoint(runtime_path)
                    runner.t_env = loaded.pop("t_env", step)
                    replay = loaded.pop("replay", None)
                    runtime.update(loaded)
                    if replay is not None:
                        buffer.data.transition_data = {key: value.to(buffer.device) for key, value in replay["transition_data"].items()}
                        buffer.data.episode_data = {key: value.to(buffer.device) for key, value in replay["episode_data"].items()}
                        buffer.buffer_index = replay["buffer_index"]
                        buffer.episodes_in_buffer = replay["episodes_in_buffer"]
                    elif not (args.evaluate or args.save_replay):
                        logger.console_logger.warning("Resuming optimizer/model/RNG state with an empty replay buffer; this is not an exact trajectory continuation")
                elif not (args.evaluate or args.save_replay):
                    logger.console_logger.warning("runtime_state.th missing; episode counters and replay start fresh")
        elif args.pretrain:
            model_path = _project_path(args.pretrain_dir) if args.pretrain_dir else PROJECT_ROOT / "pretrain_model" / args.name / args.env_args["map_name"]
            if not model_path.is_dir():
                raise FileNotFoundError("Pretrained defender not found: {}. Supply pretrain_dir or set pretrain=False.".format(model_path))
            learner.load_models(str(model_path), resume=False)

        def evaluation_runner():
            nonlocal eval_runner
            if not args.separate_eval_env or args.evaluate or args.save_replay:
                return runner
            if eval_runner is None:
                eval_args = deepcopy(args)
                eval_args.env_args["seed"] = args.seed + args.eval_seed_offset
                with isolated_rng(args.seed + args.eval_seed_offset):
                    eval_runner = runner_registry[args.runner](args=eval_args, logger=logger)
                eval_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac,
                                  qdifference_transformer=qdifference_transformer, planning_transformer=planning_transformer)
                eval_runner.setup_learner(learner)
                eval_runner.setup_mac_for_attack(mac)
            eval_runner.t_env = runner.t_env
            return eval_runner

        if args.evaluate or args.save_replay:
            _evaluate(args, evaluation_runner(), args.seed + args.eval_seed_offset)
            return
        start_time = time.monotonic()
        logger.console_logger.info("Beginning training for %s environment steps", args.t_max)
        collection_seconds = update_seconds = evaluation_seconds = 0.0
        initial_t_env = runner.t_env
        while runner.t_env < args.t_max:
            tick = time.monotonic()
            episode_batch = runner.run_wolfpack_attacker(test_mode=False)
            buffer.insert_episode_batch(episode_batch)
            collection_seconds += time.monotonic() - tick
            tick = time.monotonic()
            if buffer.can_sample(args.batch_size):
                sample = buffer.sample(args.batch_size)
                sample = sample[:, :sample.max_t_filled()]
                if str(sample.device) != args.device:
                    pin = args.replay_pin_memory and args.use_cuda and str(sample.device) == "cpu"
                    if pin:
                        sample.pin_memory()
                    sample.to(args.device, non_blocking=pin)
                learner.train(sample, runner.t_env, runtime["episode"])
            update_seconds += time.monotonic() - tick
            runtime["episode"] += args.batch_size_run
            if runner.t_env - runtime["last_test_T"] >= args.test_interval or runner.t_env >= args.t_max:
                logger.console_logger.info("t_env: %s / %s; elapsed: %s", runner.t_env, args.t_max, time_str(time.monotonic() - start_time))
                tick = time.monotonic()
                _evaluate(args, evaluation_runner(), args.seed + args.eval_seed_offset)
                evaluation_seconds += time.monotonic() - tick
                runtime["last_test_T"] = runner.t_env
            if args.save_model and (runner.t_env - runtime["model_save_time"] >= args.save_model_interval or runtime["model_save_time"] == 0 or runner.t_env >= args.t_max):
                runtime["model_save_time"] = runner.t_env
                save_path = Path(args.local_results_path) / "models" / args.unique_token / str(runner.t_env)
                save_path.mkdir(parents=True, exist_ok=True)
                learner.save_models(str(save_path))
                saved_runtime = dict(runtime, t_env=runner.t_env)
                if args.checkpoint_save_replay:
                    saved_runtime["replay"] = {"transition_data": {key: value.cpu() for key, value in buffer.data.transition_data.items()},
                                               "episode_data": {key: value.cpu() for key, value in buffer.data.episode_data.items()},
                                               "buffer_index": buffer.buffer_index, "episodes_in_buffer": buffer.episodes_in_buffer}
                th.save(saved_runtime, save_path / "runtime_state.th")
                logger.console_logger.info("Saved checkpoint to %s", save_path)
            if runner.t_env - runtime["last_log_T"] >= args.log_interval:
                logger.log_stat("episode", runtime["episode"], runner.t_env)
                logger.log_stat("throughput_env_steps_per_second", (runner.t_env - initial_t_env) / max(time.monotonic() - start_time, 1e-9), runner.t_env)
                logger.log_stat("time_collection_seconds", collection_seconds, runner.t_env)
                logger.log_stat("time_update_seconds", update_seconds, runner.t_env)
                logger.log_stat("time_evaluation_seconds", evaluation_seconds, runner.t_env)
                logger.print_recent_stats()
                runtime["last_log_T"] = runner.t_env
        logger.console_logger.info("Finished training at t_env=%s", runner.t_env)
    finally:
        try:
            runner.close_env()
        finally:
            if eval_runner is not None:
                eval_runner.close_env()

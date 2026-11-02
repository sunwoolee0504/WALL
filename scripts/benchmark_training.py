#!/usr/bin/env python3
"""Time actual optimized learner updates on reproducible synthetic episodes.

This includes TD backward, world-model backward, counterfactual planning and
Q-difference backward. It excludes SC2, collection and evaluation, and is not a
before/after benchmark. CUDA is used only when explicitly requested.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import yaml

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.basic_controller import BasicMAC
from learners.q_learner import QLearner
from modules.step.models import Planning_Transformer, Qdifference_Transformer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mixer", choices=("qmix", "vdn", "dmaq"), default="qmix")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--episode-length", type=int, default=80)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--profile-output", type=Path)
    p.add_argument("--output", type=Path)
    flags = p.parse_args()
    if min(flags.batch_size, flags.episode_length, flags.iterations, flags.threads) < 1:
        p.error("batch-size, episode-length, iterations and threads must be positive")
    if flags.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is not available in this interpreter")
    if flags.amp and flags.device != "cuda":
        p.error("--amp is supported only with --device cuda")
    torch.set_num_threads(flags.threads)
    torch.manual_seed(42)
    config = yaml.safe_load((ROOT / "src/config/default.yaml").read_text())
    name = {"qmix": "qmix", "vdn": "vdn", "dmaq": "qplex"}[flags.mixer]
    config.update(yaml.safe_load((ROOT / ("src/config/algs/wall_" + name + ".yaml")).read_text()))
    config.update(n_agents=5, n_actions=11, state_shape=120, obs_shape=80,
                  num_followup_agents=2, episode_limit=flags.episode_length,
                  learner_log_interval=1, batch_size=flags.batch_size,
                  device=flags.device, use_amp=flags.amp, compile_models=flags.compile,
                  env_args={"map_name": "synthetic_not_a_smac_map"})
    args = SimpleNamespace(**config)
    scheme = {"state": {"vshape": args.state_shape},
              "obs": {"vshape": args.obs_shape, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "forced_actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": args.n_actions, "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    preprocess = {key: (key + "_onehot", [OneHot(args.n_actions)]) for key in ("actions", "forced_actions")}
    b, t, n, a = flags.batch_size, flags.episode_length + 1, args.n_agents, args.n_actions
    batch = EpisodeBatch(scheme, {"agents": n}, b, t, preprocess=preprocess, device=flags.device)
    terminal = torch.zeros(b, t, 1, dtype=torch.uint8)
    terminal[:, -2] = 1
    batch.update({"state": torch.randn(b, t, args.state_shape), "obs": torch.randn(b, t, n, args.obs_shape),
                  "actions": torch.randint(a, (b, t, n, 1)), "forced_actions": torch.randint(a, (b, t, n, 1)),
                  "avail_actions": torch.ones(b, t, n, a, dtype=torch.int),
                  "reward": torch.randn(b, t, 1), "terminated": terminal})
    mac = BasicMAC(batch.scheme, {"agents": n}, args)
    learner = QLearner(mac, batch.scheme, None, args, Qdifference_Transformer(args), Planning_Transformer(args)).to(flags.device)
    def sync():
        if flags.device == "cuda":
            torch.cuda.synchronize()
    learner.train(batch, 1, 1)
    sync()
    latencies = []
    for i in range(flags.iterations):
        start = time.perf_counter()
        metrics = learner.train(batch, i + 2, i + 2)
        sync()
        latencies.append(1000 * (time.perf_counter() - start))
    result = {"kind": "optimized_full_learner_synthetic_update", "python": platform.python_version(),
              "torch": torch.__version__, "device": flags.device, "threads": flags.threads,
              "mixer": flags.mixer, "dimensions": {"batch": b, "transitions": t-1, "agents": n,
              "actions": a, "state": args.state_shape, "observation": args.obs_shape,
              "context": args.trans_input_len, "attack_horizon": args.attack_period},
              "amp": flags.amp, "compile": flags.compile, "warmup_updates": 1,
              "iterations": flags.iterations, "update_median_ms": statistics.median(latencies),
              "update_samples_ms": latencies, "last_losses": {k: v for k,v in metrics.items() if k.startswith("loss")},
              "limitations": "Synthetic transitions; no collection/SC2/evaluation; not baseline comparison or convergence result."}
    if flags.profile_output:
        flags.profile_output.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if flags.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            learner.train(batch, flags.iterations+2, flags.iterations+2)
            sync()
        prof.export_chrome_trace(str(flags.profile_output))
        result["profile_trace"] = str(flags.profile_output)
    if flags.output:
        flags.output.parent.mkdir(parents=True, exist_ok=True)
        flags.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

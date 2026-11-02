#!/usr/bin/env python3
"""Controlled synthetic benchmark against the preserved sibling implementation.

Example: python scripts/benchmark_models.py --threads 1 --iterations 10 \
    --baseline-dir /path/to/original-wall --output model_benchmark.json

Predictor architectures/targets differ, so these are component timings, not an
end-to-end training speedup or a learning-quality comparison. The default single
Transformer layer matches the paper preset. Policy parameters are identical.
"""
import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Importing the preserved baseline must not write caches into the source folder.
sys.dont_write_bytecode = True

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.agents.rnn_agent import RNNAgent
from modules.step.models import Planning_Transformer, Qdifference_Transformer


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError("Baseline source file missing: %s. Supply --baseline-dir with the original WALL/PyMARL source checkout; the public release does not include it." % path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", "--original-root", dest="baseline_dir", type=Path, required=True,
                        help="Original WALL/PyMARL source checkout for comparison (not included in the public release)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--policy-length", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    opts = parser.parse_args()
    if min(opts.threads, opts.iterations, opts.layers, opts.heads, opts.batch_size, opts.context, opts.policy_length) < 1 or opts.warmup < 0:
        parser.error("Sizes/iterations/threads must be positive; warmup must be nonnegative")
    if 64 % opts.heads:
        parser.error("--heads must divide the fixed width 64")
    if opts.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(opts.threads)
    torch.manual_seed(42)
    device = torch.device(opts.device)
    try:
        old_models = load_module("wall_benchmark_original_models", opts.baseline_dir / "src/modules/step/models.py")
        old_agents = load_module("wall_benchmark_original_agents", opts.baseline_dir / "src/modules/agents/rnn_agent.py")
    except FileNotFoundError as exc:
        parser.error(str(exc))
    config = SimpleNamespace(
        state_shape=168, obs_shape=80, n_agents=8, n_actions=14, trans_embed_dim=64,
        num_trans_layer=opts.layers, trans_input_len=opts.context, num_trans_head=opts.heads,
        attack_period=20, trans_dropout=0.1, trans_ff_multiplier=4,
        attention_backend="sdpa", env_max_timestep=max(200, opts.context + 1), rnn_hidden_dim=64)
    b, t = opts.batch_size, opts.context
    state = torch.randn(b, t, config.state_shape, device=device)
    obs = torch.randn(b, t, config.n_agents, config.obs_shape, device=device)
    history = torch.randn(b, t, config.n_agents, config.rnn_hidden_dim, device=device)
    actions = torch.randn(b, t, config.n_agents, config.n_actions, device=device)
    times = torch.arange(t, device=device).expand(b, -1)
    valid = torch.ones(b, t, device=device, dtype=torch.bool)
    ids = torch.eye(config.n_agents, device=device)[None, None].expand(b, t, -1, -1)
    old_planner = old_models.Planning_Transformer(config).to(device).eval()
    planner = Planning_Transformer(config).to(device).eval()
    old_qdiff = old_models.Qdifference_Transformer(config).to(device).eval()
    qdiff = Qdifference_Transformer(config).to(device).eval()

    @torch.no_grad()
    def old_plan():
        next_state, obs_embed = old_planner(state, history, actions)
        return next_state, torch.stack([
            old_planner.get_next_obs(obs_embed, ids[:, :, i]) for i in range(config.n_agents)], dim=2)

    @torch.no_grad()
    def plan():
        return planner(state, obs, actions, times, valid, return_aux=True)

    @torch.no_grad()
    def old_q():
        return torch.stack([old_qdiff(times, state, ids[:, :, i]) for i in range(config.n_agents)], dim=2)

    @torch.no_grad()
    def q():
        return qdiff(times, state, valid)

    input_dim = config.obs_shape + config.n_actions + config.n_agents
    old_rnn = old_agents.RNNAgent(input_dim, config).to(device)
    rnn = RNNAgent(input_dim, config).to(device)
    rnn.load_state_dict(old_rnn.state_dict())
    seq = torch.randn(b * config.n_agents, opts.policy_length, input_dim, device=device)
    hidden = torch.zeros(b * config.n_agents, config.rnn_hidden_dim, device=device)

    def old_train():
        old_rnn.zero_grad(set_to_none=True)
        recurrent = hidden
        outputs = []
        for step in range(seq.shape[1]):
            output, recurrent = old_rnn(seq[:, step], recurrent)
            outputs.append(output)
        torch.stack(outputs, dim=1).square().mean().backward()

    def train():
        rnn.zero_grad(set_to_none=True)
        rnn.forward_sequence(seq, hidden)[0].square().mean().backward()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(fn):
        for _ in range(opts.warmup):
            fn()
        synchronize()
        samples = []
        for _ in range(opts.iterations):
            start = time.perf_counter()
            fn()
            synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return {"median_ms": statistics.median(samples), "min_ms": min(samples),
                "max_ms": max(samples), "samples_ms": samples}

    timings = {}
    for name, fn in (("planner_original", old_plan), ("planner_optimized", plan),
                     ("qdiff_all_agents_original", old_q), ("qdiff_expected_effect_optimized", q),
                     ("policy_training_original", old_train), ("policy_training_optimized", train)):
        timings[name] = measure(fn)
    loaded = []
    for path in sorted((ROOT / "pretrain_model").glob("*/*/agent.th")):
        weights = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_config = SimpleNamespace(rnn_hidden_dim=weights["fc1.weight"].shape[0], n_actions=weights["fc2.weight"].shape[0])
        defender = RNNAgent(weights["fc1.weight"].shape[1], checkpoint_config)
        defender.load_state_dict(weights, strict=True)
        loaded.append(str(path.relative_to(ROOT)))
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "threads": torch.get_num_threads(), "iterations": opts.iterations, "warmup": opts.warmup,
        "seed": 42, "batch_size": b, "context": t, "policy_length": opts.policy_length,
        "configuration": vars(config), "timings": timings,
        "parameters": {"planner_original": sum(p.numel() for p in old_planner.parameters()),
                       "planner_optimized": sum(p.numel() for p in planner.parameters()),
                       "qdiff_original": sum(p.numel() for p in old_qdiff.parameters()),
                       "qdiff_optimized": sum(p.numel() for p in qdiff.parameters())},
        "real_legacy_policy_checkpoints_loaded": loaded,
        "limits": ["Synthetic component timings; no environment step, full learner, or win-rate measurement.",
                   "Predictor architectures and targets differ: old planner uses hidden states; optimized uses raw observations and adds auxiliary heads.",
                   "Original Q-difference repeats the trunk for every agent; optimized predicts expected effect once.",
                   "Policy training uses identical weights and includes forward and backward, excluding optimizer.step.",
                   "No claim of end-to-end training speedup or unchanged predictor quality."],
    }
    encoded = json.dumps(result, indent=2)
    if opts.output:
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a final checkpoint with optional independent frozen attack predictors."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Saved Sacred config.json for the defender")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Defender learner.th checkpoint")
    parser.add_argument("--scheduler-checkpoint", type=Path,
                        help="Independent attacker learner.th; only predictor weights are used")
    parser.add_argument("--attacker-seed", type=int,
                        help="Attacker training seed, for experiment metadata")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num-attack-test", type=int)
    parser.add_argument("--env-seed", type=int, default=1931001)
    parser.add_argument("--attack-seed", type=int, default=1932001)
    parser.add_argument("--sc2path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New result directory; never overwritten")
    parser.add_argument("--modes", nargs="+", default=["natural", "wolfpack"],
                        choices=["natural", "wolfpack", "uniform", "random_followup"])
    args = parser.parse_args()
    if args.episodes < 1 or (args.num_attack_test is not None and args.num_attack_test < 0):
        parser.error("Episode count must be positive and attack budget nonnegative")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Evaluation modes must not be repeated")
    config = json.loads(args.config.read_text())
    learned_timing = bool(set(args.modes) & {"wolfpack", "random_followup"})
    if (not config.get("train_attack_predictors", True)
            and learned_timing and args.scheduler_checkpoint is None):
        parser.error("This defender has untrained attack predictors; supply a trained "
                     "--scheduler-checkpoint for attacked evaluation")
    if args.num_attack_test is not None:
        config["num_attack_test"] = args.num_attack_test
    checkpoints = [args.checkpoint]
    if args.scheduler_checkpoint is not None:
        checkpoints.append(args.scheduler_checkpoint)
    for path in checkpoints:
        if not path.is_file():
            parser.error("Missing checkpoint: " + str(path))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    task = dict(
        id="checkpoint_evaluation", source=str(Path(__file__).resolve().parents[1]),
        config=str(config_path), config_sha256=digest(config_path),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        map=config["env_args"]["map_name"], seed=config["seed"], episodes=args.episodes,
        env_seed=args.env_seed, attack_seed=args.attack_seed,
        sc2path=str(args.sc2path.resolve()), modes=args.modes,
        prediction_episodes=0, output=str(output / "result.json"),
    )
    if args.scheduler_checkpoint is not None:
        task.update(scheduler_checkpoint=str(args.scheduler_checkpoint.resolve()),
                    scheduler_sha256=digest(args.scheduler_checkpoint),
                    attacker_seed=args.attacker_seed)
    task_path = output / "task.json"
    task_path.write_text(json.dumps(task, indent=2) + "\n")
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit_saved_models.py")),
                    "--task", str(task_path)], check=True)
    result = json.loads((output / "result.json").read_text())
    if result["status"] != "completed":
        raise RuntimeError("Evaluation did not complete")
    for mode in args.modes:
        rows = [r for r in result["episodes"] if r["mode"] == mode]
        if len(rows) != args.episodes:
            raise RuntimeError("Unexpected episode count for " + mode)
        wins = sum(r["win"] for r in rows)
        print(mode, "%d/%d wins (%.2f%%)" % (wins, len(rows), 100 * wins / len(rows)))
    print("Saved", output / "result.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate portable release models under the same frozen attacker and budget.

selected: released single defender, natural/attacked, and pretrained QMIX.
five-seed: all five defenders, natural/attacked, and pretrained QMIX.
cross: all five defenders x five attackers plus natural and QMIX diagnostics.
All evaluation workers run on CPU; the output directory is never overwritten.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from release_utils import (
    ROOT,
    load_catalog,
    safe_path,
    verified_asset,
    read_json,
    save_json,
    digest,
)


def build_tasks(
    catalog,
    maps,
    asset_root,
    output,
    profile,
    episodes,
    sc2path,
    env_seed=2731001,
    attack_seed=2732001,
):
    tasks = []
    verified = {}

    def asset(path):
        if path not in verified:
            verified[path] = verified_asset(asset_root, path, catalog)
        return verified[path]

    for name in maps:
        entry = catalog["maps"][name]
        seeds = (
            [entry["selected_defender_seed"]]
            if profile == "selected"
            else list(range(1, 6))
        )
        attackers = (
            list(range(1, 6))
            if profile == "cross"
            else [entry["selected_attacker_seed"]]
        )
        conditions = [("wall", s, a, "wolfpack") for s in seeds for a in attackers]
        conditions += [
            ("wall", s, entry["selected_attacker_seed"], "natural") for s in seeds
        ]
        conditions += [
            ("pretrained_qmix", entry["selected_defender_seed"], a, "wolfpack")
            for a in attackers
        ]
        for defender, seed, a, mode in conditions:
            model = entry["defenders"][str(seed)]
            attacker = entry["attackers"][str(a)]
            cp, ap = asset(model["checkpoint"]), asset(attacker["checkpoint"])
            cfg = read_json(safe_path(ROOT, model["config"]))
            cfg.update(entry["evaluation"])
            tid = "%s_%s_d%d_a%d_%s" % (name, defender, seed, a, mode)
            path = output / "configs" / (tid + ".json")
            save_json(path, cfg)
            task = dict(
                id=tid,
                source=str(ROOT),
                config=str(path),
                config_sha256=digest(path),
                checkpoint=str(cp),
                checkpoint_sha256=digest(cp),
                scheduler_checkpoint=str(ap),
                scheduler_sha256=digest(ap),
                map=name,
                seed=seed,
                defender=defender,
                attacker_seed=a,
                sc2path=str(sc2path.resolve()),
                episodes=episodes,
                prediction_episodes=0,
                modes=[mode],
                env_seed=env_seed + 10000 * catalog["map_order"].index(name),
                attack_seed=attack_seed + 10000 * catalog["map_order"].index(name),
                output=str(output / "results" / (tid + ".json")),
                **entry["evaluation"]
            )
            if defender == "pretrained_qmix":
                for filename in ("agent.th", "mixer.th"):
                    asset(entry["pretrained_directory"] + "/" + filename)
                task["vanilla"] = str(
                    safe_path(asset_root, entry["pretrained_directory"])
                )
            save_json(output / "tasks" / (tid + ".json"), task)
            tasks.append(task)
    return tasks


def run_tasks(tasks, output, workers):
    queue = list(tasks)
    active = []
    completed = 0
    status = dict(stage="running", completed=0, total=len(tasks), pid=os.getpid())

    def stop(*_):
        raise KeyboardInterrupt()

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        while queue or active:
            while queue and len(active) < workers:
                task = queue.pop(0)
                log = (output / "logs" / (task["id"] + ".log")).open("wb")
                p = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts/audit_saved_models.py"),
                        "--task",
                        str(output / "tasks" / (task["id"] + ".json")),
                    ],
                    cwd=ROOT,
                    env=dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES="",
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active.append((p, log, task))
            for item in active[:]:
                p, log, task = item
                code = p.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                if code:
                    raise RuntimeError(
                        "Evaluation failed: " + task["id"] + "; see its log"
                    )
                result = read_json(task["output"])
                if (
                    result.get("status") != "completed"
                    or result["task"] != task
                    or result["checkpoint_sha256"] != task["checkpoint_sha256"]
                    or result["scheduler_sha256"] != task["scheduler_sha256"]
                    or len(result["episodes"]) != task["episodes"]
                ):
                    raise RuntimeError(
                        "Incomplete or mismatched evaluation: " + task["id"]
                    )
                completed += 1
            status.update(
                completed=completed,
                pending=len(queue),
                active=[t["id"] for _, _, t in active],
            )
            save_json(output / "status.json", status)
            if active:
                time.sleep(0.25)
        status.update(stage="completed", completed=len(tasks), active=[], pending=0)
    except BaseException as exc:
        status.update(stage="failed", error=repr(exc))
        raise
    finally:
        for p, log, _ in active:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            log.close()
        signal.signal(signal.SIGTERM, previous)
        save_json(output / "status.json", status)


def summarize(tasks, output):
    cells = []
    groups = defaultdict(list)
    for task in tasks:
        rows = read_json(task["output"])["episodes"]
        if any(not 0 <= r["changed_steps"] <= task["num_attack_test"] for r in rows):
            raise ValueError("Attack budget exceeded")
        p = statistics.mean(r["win"] for r in rows)
        n = len(rows)
        z = 1.959963984540054
        mid = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        row = dict(
            map=task["map"],
            defender=task["defender"],
            seed=task["seed"],
            attacker_seed=task["attacker_seed"],
            mode=task["modes"][0],
            episodes=n,
            win_pct=100 * p,
            wilson95_pct=[100 * (mid - half), 100 * (mid + half)],
            changed_steps=statistics.mean(r["changed_steps"] for r in rows),
            num_attack_test=task["num_attack_test"],
            num_followup_agents=task["num_followup_agents"],
            temperature=task["temperature"],
        )
        cells.append(row)
        groups[(row["map"], row["defender"], row["mode"])].append(row)
    metrics = []
    for (name, defender, mode), rows in groups.items():
        per_seed = defaultdict(list)
        for row in rows:
            per_seed[row["seed"]].append(row["win_pct"])
        values = [statistics.mean(v) for v in per_seed.values()]
        metrics.append(
            dict(
                map=name,
                defender=defender,
                mode=mode,
                mean_win_pct=statistics.mean(values),
                defender_seed_sd_pp=statistics.stdev(values)
                if len(values) > 1
                else None,
                n_defender_seeds=len(values),
                note="Pretrained QMIX is one policy; attack repetitions are not independent defender training seeds."
                if defender == "pretrained_qmix"
                else "Mean across the requested defenders and attacker bank.",
            )
        )
    summary = dict(cells=cells, metrics=metrics)
    save_json(output / "summary.json", summary)
    for row in metrics:
        print(row["map"], row["defender"], row["mode"], "%.2f%%" % row["mean_win_pct"])
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="all")
    p.add_argument(
        "--profile", choices=["selected", "five-seed", "cross"], default="selected"
    )
    p.add_argument("--asset-root", type=Path, default=ROOT)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--sc2path",
        type=Path,
        default=Path(os.environ.get("SC2PATH", Path.home() / "StarCraftII")),
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--env-seed", type=int, default=2731001)
    p.add_argument("--attack-seed", type=int, default=2732001)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    catalog = load_catalog()
    if a.verify_only:
        for path in catalog["assets"]:
            verified_asset(a.asset_root, path, catalog)
        print("Verified %d release assets." % len(catalog["assets"]))
        return
    if a.map != "all" and a.map not in catalog["maps"]:
        p.error("Unknown map")
    if not 1 <= a.workers <= 8 or a.episodes < 1:
        p.error("workers must be 1..8; episodes positive")
    if a.output is None:
        p.error("--output is required for evaluation")
    if not a.dry_run and not (a.sc2path / "Versions").is_dir():
        p.error("SC2PATH must contain Versions")
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for child in ("configs", "tasks", "logs", "results"):
        (output / child).mkdir()
    maps = catalog["map_order"] if a.map == "all" else [a.map]
    tasks = build_tasks(
        catalog,
        maps,
        a.asset_root,
        output,
        a.profile,
        a.episodes,
        a.sc2path,
        a.env_seed,
        a.attack_seed,
    )
    save_json(
        output / "manifest.json",
        dict(
            tasks=[t["id"] for t in tasks],
            catalog_sha256=digest(ROOT / "src/config/release_catalog.json"),
            profile=a.profile,
            episodes=a.episodes,
            worker_sha256=digest(ROOT / "scripts/audit_saved_models.py"),
        ),
    )
    if a.dry_run:
        print("Prepared %d tasks; no evaluations launched." % len(tasks))
        return
    run_tasks(tasks, output, a.workers)
    summarize(tasks, output)


if __name__ == "__main__":
    main()

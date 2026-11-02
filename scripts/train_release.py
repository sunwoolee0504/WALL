#!/usr/bin/env python3
"""Train the documented per-map release cohort from the supplied pretrained QMIX."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from release_utils import ROOT, CATALOG, load_catalog, safe_path, verified_asset


def command_for(
    map_name, seed, catalog, asset_root, output, cpu=False, steps=3000000, overrides=()
):
    entry = catalog["maps"][map_name]
    pretrain = safe_path(asset_root, entry["pretrained_directory"])
    for name in ("agent.th", "mixer.th", "opt.th"):
        verified_asset(asset_root, entry["pretrained_directory"] + "/" + name, catalog)
    config = safe_path(ROOT, entry["training_config"])
    if not config.is_file():
        raise FileNotFoundError(config)
    return [
        sys.executable,
        str(ROOT / "src/main.py"),
        "--config=wall_qmix",
        "--env-config=sc2",
        "with",
        str(config),
        "env_args.map_name=" + map_name,
        "seed=" + str(seed),
        "t_max=" + str(steps),
        "pretrain=True",
        "pretrain_dir=" + str(pretrain),
        "use_cuda=" + str(not cpu),
        "local_results_path=" + str(Path(output).resolve()),
        "label=release_%s_seed%d" % (map_name, seed),
        *overrides,
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--asset-root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=ROOT / "results/release_training")
    p.add_argument("--t-max", type=int, default=3000000)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Explicit Sacred key=value; repeatable, recorded in run config",
    )
    a = p.parse_args()
    catalog = load_catalog()
    if a.map not in catalog["maps"]:
        p.error("Unknown map; choose " + ", ".join(catalog["map_order"]))
    if a.t_max < 1 or a.seed < 0:
        p.error("t-max must be positive; seed must be nonnegative")
    if any("=" not in item for item in a.override):
        p.error("--override requires key=value")
    cmd = command_for(
        a.map, a.seed, catalog, a.asset_root, a.output, a.cpu, a.t_max, a.override
    )
    if a.dry_run:
        print(json.dumps(cmd, indent=2))
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

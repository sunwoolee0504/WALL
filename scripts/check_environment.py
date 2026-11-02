#!/usr/bin/env python3
"""Read-only diagnostics: no simulator launch, installation or GPU allocation."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import envs  # Resolves a default SC2PATH without importing SMAC.
    import torch
    report = {
        "python": sys.version.split()[0], "interpreter": sys.executable,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "sc2_path": os.environ.get("SC2PATH"),
    }
    for name in ("numpy", "PyYAML", "sacred", "pysc2", "s2clientprotocol", "protobuf", "wandb"):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report[name] = None
    try:
        # Resolving the package identifies unrelated AutoML installations early.
        spec = importlib.util.find_spec("smac.env")
        report["smac_env"] = spec.origin if spec else None
        if spec is not None:
            from smac.env import StarCraft2Env
            report["smac_import"] = "ok"
        else:
            report["smac_import"] = "missing; install oxwhirl/smac, not PyPI smac"
    except Exception as exc:
        report["smac_import"] = "{}: {}".format(type(exc).__name__, exc)
    try:
        report["smac_provenance"] = json.loads(importlib.metadata.distribution("SMAC").read_text("direct_url.json") or "null")
    except importlib.metadata.PackageNotFoundError:
        report["smac_provenance"] = None
    sc2_path = Path(report["sc2_path"]) if report["sc2_path"] else None
    report["sc2_directory_exists"] = bool(sc2_path and sc2_path.is_dir())
    report["smac_maps_exists"] = bool(sc2_path and (sc2_path / "Maps" / "SMAC_Maps").is_dir())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

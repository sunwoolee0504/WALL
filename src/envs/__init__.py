"""Environment registry that remains importable without StarCraft II/SMAC."""
import os
from pathlib import Path
import sys


def make_sc2(**kwargs):
    try:
        from smac.env import StarCraft2Env
    except ImportError as exc:
        raise RuntimeError(
            "SC2 training requires SMAC from https://github.com/oxwhirl/smac "
            "and a StarCraft II installation; set SC2PATH to its directory."
        ) from exc
    return StarCraft2Env(**kwargs)


REGISTRY = {"sc2": make_sc2}

if sys.platform == "linux" and "SC2PATH" not in os.environ:
    project = Path(__file__).resolve().parents[2]
    candidates = (project / "3rdparty" / "StarCraftII", Path.home() / "StarCraftII")
    for candidate in candidates:
        if candidate.is_dir():
            os.environ["SC2PATH"] = str(candidate)
            break

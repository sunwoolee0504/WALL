"""Exercise shell entry points with stand-in executables: no pip or network."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, body):
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)


@pytest.fixture
def setup_workspace(tmp_path):
    project = tmp_path / "wall project"
    fake_bin = tmp_path / "fake bin"
    project.mkdir()
    fake_bin.mkdir()
    for relative in ("scripts/install_sc2.sh", "scripts/install_optimized_env.sh", "requirements.txt"):
        destination = project / relative
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    sc2 = tmp_path / "StarCraft II"
    (sc2 / "Versions").mkdir(parents=True)
    (sc2 / "Maps/SMAC_Maps").mkdir(parents=True)
    capture = tmp_path / "commands.jsonl"
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
               SC2PATH=str(sc2), WALL_SETUP_CAPTURE=str(capture))
    return project, fake_bin, capture, env


def capture_executable(path):
    executable(path, 'import os, sys, json\nwith open(os.environ["WALL_SETUP_CAPTURE"], "a") as stream:\n    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n')


def calls(capture):
    return [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []


def test_setup_rejects_invalid_arguments_before_external_commands(setup_workspace, tmp_path):
    project, fake_bin, capture, env = setup_workspace
    capture_executable(fake_bin / "curl")
    for script, args in (("scripts/install_optimized_env.sh", ["cuda9"]),
                         ("scripts/install_optimized_env.sh", ["cpu", str(tmp_path)])):
        result = subprocess.run(["bash", str(project / script), *args], env=env, capture_output=True, text=True)
        assert result.returncode == 2, result.stderr
    assert calls(capture) == []


def test_installer_selects_wheel_then_uses_canonical_dependencies(setup_workspace, tmp_path):
    project, fake_bin, capture, env = setup_workspace
    bootstrap = fake_bin / "python-bootstrap"
    stub = '''import json, os, sys
with open(os.environ["WALL_SETUP_CAPTURE"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:] == ["-m", "pip", "freeze"]:
    print("torch==2.5.1+cu121")
'''
    stub_source = "#!" + sys.executable + "\n" + stub
    executable(bootstrap, f'''import pathlib, subprocess, sys
if sys.argv[1:3] == ["-m", "venv"]:
    target = pathlib.Path(sys.argv[3]) / "bin" / "python"
    target.parent.mkdir(parents=True)
    target.write_text({stub_source!r})
    target.chmod(0o755)
else:
    raise SystemExit(subprocess.run([{sys.executable!r}] + sys.argv[1:]).returncode)
''')
    env["WALL_PYTHON_BIN"] = str(bootstrap)
    subprocess.run(["bash", str(project / "scripts/install_optimized_env.sh"), "cu121", "new environment"], cwd=tmp_path, env=env, check=True, capture_output=True)
    recorded = calls(capture)
    wheel = recorded[1]
    assert wheel[:4] == ["-m", "pip", "install", "torch==2.5.1"]
    assert wheel[-1] == "https://download.pytorch.org/whl/cu121"
    assert recorded[2] == ["-m", "pip", "install", "-r", str(project / "requirements.txt")]
    assert recorded[3] == ["-m", "pip", "check"]
    assert (tmp_path / "new environment/requirements-resolved.txt").read_text().strip() == "torch==2.5.1+cu121"


def test_sc2_installer_keeps_existing_installation_without_network(setup_workspace):
    project, fake_bin, capture, env = setup_workspace
    capture_executable(fake_bin / "curl")
    subprocess.run(["bash", str(project / "scripts/install_sc2.sh")], env=env, check=True, capture_output=True)
    assert calls(capture) == []


def test_sc2_installer_uses_requested_directory_and_preserves_sources(setup_workspace, tmp_path):
    project, fake_bin, capture, env = setup_workspace
    game_zip = tmp_path / "game.zip"
    maps_zip = tmp_path / "maps.zip"
    with zipfile.ZipFile(game_zip, "w") as archive:
        archive.writestr("StarCraftII/Versions/Base75689/SC2_x64", "stand-in executable")
    with zipfile.ZipFile(maps_zip, "w") as archive:
        archive.writestr("SMAC_Maps/3m.SC2Map", "stand-in map")
    executable(fake_bin / "curl", f'''import pathlib, shutil, sys
args = sys.argv[1:]
source = {str(game_zip)!r} if args[-1].endswith("SC2.4.10.zip") else {str(maps_zip)!r}
assert args[-1].startswith("https://")
shutil.copyfile(source, args[args.index("--output") + 1])
''')
    destination = tmp_path / "new game location"
    subprocess.run(["bash", str(project / "scripts/install_sc2.sh"), str(destination)], env=env, check=True, capture_output=True)
    assert (destination / "Versions/Base75689/SC2_x64").is_file()
    assert (destination / "Maps/SMAC_Maps/3m.SC2Map").is_file()
    assert game_zip.exists() and maps_zip.exists()

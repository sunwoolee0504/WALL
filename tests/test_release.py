"""Release export keeps local research artifacts private and source untouched."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_github_release.py"
spec = importlib.util.spec_from_file_location("wall_release_export", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def write(root, relative, content="sample\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def source_tree(tmp_path):
    root = tmp_path / "source"
    for name in ("README.md", "LICENSE", "requirements-optimized.txt", ".gitignore", ".gitattributes",
                 "CITATION.cff", "install_sc2.sh", "run.sh", "run_interactive.sh", "src/main.py", "src/config/default.yaml", "tests/test_sample.py",
                 "scripts/install.sh", "docs/training.md", "docker/Dockerfile", ".github/workflows/test.yml"):
        write(root, name)
    (root / "scripts/install.sh").chmod(0o755)
    for name in ("WALL_paper.pdf", "OPTIMIZATION_NOTES_KO.md", "WALL_IMPLEMENTATION_REVIEW_KO.md",
                 "pretrain_model/3m/agent.th", "results/run.json", "review_artifacts/local.json",
                 "validation/private.json", ".git/config", ".env", "src/.env.production",
                 "src/credentials.json", "src/secrets.yaml", "src/key.pem", "src/model.th",
                 "src/__pycache__/main.pyc", "tests/.pytest_cache/cache.json", "scripts/.venv/config.py",
                 "scripts/results/debug.json", "docs/validation/local.json"):
        write(root, name, "must stay local\n")
    return root


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_directory_export_is_standalone_allowlisted_and_preserves_source(tmp_path):
    source = source_tree(tmp_path)
    before = snapshot(source)
    destination = tmp_path / "public"
    result = release.prepare_release(source, destination)
    expected = {"README.md", "LICENSE", "requirements-optimized.txt", ".gitignore", ".gitattributes",
                "CITATION.cff", "install_sc2.sh", "run.sh", "run_interactive.sh", "src/main.py", "src/config/default.yaml", "tests/test_sample.py",
                "scripts/install.sh", "docs/training.md", "docker/Dockerfile", ".github/workflows/test.yml",
                release.MANIFEST_NAME}
    assert set(snapshot(destination)) == expected
    assert snapshot(source) == before
    assert result["pretrained_binary_weights_included"] is False
    assert (destination / "scripts/install.sh").stat().st_mode & 0o111
    manifest = json.loads((destination / release.MANIFEST_NAME).read_text())
    for entry in manifest["files"]:
        content = (destination / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
        assert entry["bytes"] == len(content)
    assert str(source) not in (destination / release.MANIFEST_NAME).read_text()


def test_zip_has_identical_content_and_reproducible_bytes(tmp_path):
    source = source_tree(tmp_path)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    release.prepare_release(source, first, "zip")
    release.prepare_release(source, second, "zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert "src/main.py" in archive.namelist()
        assert "pretrain_model/3m/agent.th" not in archive.namelist()
        assert "OPTIMIZATION_NOTES_KO.md" not in archive.namelist()
        assert all(not path.startswith("/") and ".." not in Path(path).parts for path in archive.namelist())
        assert archive.getinfo("scripts/install.sh").external_attr >> 16 & 0o111
        manifest = json.loads(archive.read(release.MANIFEST_NAME))
        assert all(hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
                   for entry in manifest["files"])


@pytest.mark.parametrize("format_name", ["directory", "zip"])
def test_existing_destination_is_never_overwritten(tmp_path, format_name):
    source = source_tree(tmp_path)
    destination = write(tmp_path, "keep", "original data")
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.prepare_release(source, destination, format_name)
    assert destination.read_text() == "original data"
    directory = tmp_path / "keep-directory"
    write(directory, "sentinel", "original directory")
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.prepare_release(source, directory, format_name)
    assert (directory / "sentinel").read_text() == "original directory"


@pytest.mark.parametrize("relative,is_directory", [("src/external.py", False), ("src/external", True), ("README.md", False)])
def test_source_symlinks_cannot_copy_external_content(tmp_path, relative, is_directory):
    source = source_tree(tmp_path)
    outside = tmp_path / "outside"
    if is_directory:
        write(outside, "secret.py", "external content")
    else:
        outside.write_text("external content")
    link = source / relative
    if link.exists():
        link.unlink()
    link.symlink_to(outside, target_is_directory=is_directory)
    destination = tmp_path / "public"
    with pytest.raises(release.ReleaseError, match="symlinks"):
        release.prepare_release(source, destination)
    assert not destination.exists()
    assert link.is_symlink()


def test_broken_destination_symlink_and_source_tree_recursion_are_rejected(tmp_path):
    source = source_tree(tmp_path)
    destination = tmp_path / "broken-link"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.prepare_release(source, destination)
    assert destination.is_symlink()
    with pytest.raises(release.ReleaseError, match="outside exported"):
        release.prepare_release(source, source / "src" / "public")
    assert not (source / "src" / "public").exists()


def test_cli_requires_explicit_destination():
    result = subprocess.run([sys.executable, "-B", str(SCRIPT)], text=True, capture_output=True)
    assert result.returncode == 2
    assert "--destination" in result.stderr

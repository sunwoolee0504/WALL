#!/usr/bin/env python3
"""Export a standalone source release without changing this research workspace.

Examples (the destination must not already exist):
  python scripts/prepare_github_release.py --destination /tmp/wall-public
  python scripts/prepare_github_release.py --destination /tmp/wall-public.zip --format zip

Only explicit source/documentation paths are exported. Pretrained binary weights,
PDFs, results, local validation/review artifacts, environments, caches and known
secret file types are excluded. Review the resulting files before publishing;
filename filtering is not a substitute for inspecting source for embedded secrets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import zipfile

sys.dont_write_bytecode = True

ALLOWED_DIRECTORIES = ("src", "tests", "scripts", "docs", "docker", ".github")
ALLOWED_ROOT_FILES = frozenset({
    "README.md", "README_KO.md", "LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.md",
    "CITATION.cff", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md",
    "requirements.txt", "requirements-optimized.txt", "requirements-dev.txt",
    "pyproject.toml", "setup.py", "setup.cfg", "environment.yml", "environment.yaml",
    "MANIFEST.in", "Makefile", ".gitignore", ".gitattributes", ".dockerignore",
    "install_sc2.sh", "run.sh", "run_interactive.sh",
})
ALLOWED_SUFFIXES = frozenset({
    ".py", ".pyi", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".sh", ".bash", ".txt", ".md", ".rst", ".in", ".cff", ".csv",
})
ALLOWED_EXTENSIONLESS = frozenset({"Dockerfile", "Makefile", "LICENSE", "NOTICE"})
EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".svn", ".hg", ".codex", ".agents", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    ".ipynb_checkpoints", "node_modules", "build", "dist", "results", "logs",
    "wandb", "pretrain_model", "pretrained_models", "checkpoints", "weights",
    "review_artifacts", "validation",
})
PRIVATE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"})
PRIVATE_NAMES = frozenset({
    "credentials", "credentials.json", "credentials.yaml", "credentials.yml",
    "secrets", "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml",
    "id_rsa", "id_ed25519", "id_ecdsa", ".netrc", ".pypirc", ".npmrc",
})
MANIFEST_NAME = "RELEASE_MANIFEST.json"


class ReleaseError(ValueError):
    """The requested release cannot be created without violating export rules."""


def _excluded_name(name):
    lower = name.lower()
    return (lower in EXCLUDED_DIRECTORIES or lower in PRIVATE_NAMES
            or lower == ".env" or lower.startswith(".env.")
            or lower.startswith(".venv") or lower.endswith(tuple(PRIVATE_SUFFIXES)))


def _read_source(path, source):
    if path.is_symlink():
        raise ReleaseError("Source symlinks are not exported: %s" % path.relative_to(source))
    try:
        path.resolve(strict=True).relative_to(source)
    except (ValueError, OSError) as exc:
        raise ReleaseError("Source path escapes the selected project: %s" % path) from exc
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseError("Only regular source files may be exported: %s" % path.relative_to(source))
    mode = 0o755 if info.st_mode & 0o111 else 0o644
    return path.read_bytes(), mode


def collect_sources(source):
    """Read the allowlisted files without following symlinks or changing source."""
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ReleaseError("Source must be a project directory")
    files = {}
    for name in sorted(ALLOWED_ROOT_FILES):
        path = source / name
        if path.exists() or path.is_symlink():
            files[name] = _read_source(path, source)
    for directory in ALLOWED_DIRECTORIES:
        base = source / directory
        if base.is_symlink():
            raise ReleaseError("Source symlinks are not exported: %s" % directory)
        if not base.exists():
            continue
        if not base.is_dir():
            raise ReleaseError("Expected source directory: %s" % directory)
        for current, subdirs, names in os.walk(base, followlinks=False):
            current = Path(current)
            subdirs[:] = sorted(name for name in subdirs if not _excluded_name(name) and not name.startswith("."))
            for name in subdirs:
                if (current / name).is_symlink():
                    raise ReleaseError("Source symlinks are not exported: %s" % (current / name).relative_to(source))
            for name in sorted(names):
                if _excluded_name(name) or name.startswith("."):
                    continue
                path = current / name
                if path.suffix.lower() not in ALLOWED_SUFFIXES and name not in ALLOWED_EXTENSIONLESS:
                    continue
                relative = path.relative_to(source).as_posix()
                files[relative] = _read_source(path, source)
    if "README.md" not in files or "LICENSE" not in files:
        raise ReleaseError("The source release must include README.md and LICENSE")
    return dict(sorted(files.items()))


def _manifest(files):
    payload = {
        "format_version": 1,
        "scope": "Standalone source release; binary pretrained weights are not included.",
        "excluded": ["binary weights and PDFs", "results and local review/validation artifacts",
                     "VCS history, environments, caches and known secret filenames/types"],
        "files": [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for path, (data, _) in files.items()],
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def prepare_release(source, destination, output_format="directory"):
    """Create a new directory/zip, refusing overwrites and source-tree recursion."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if output_format not in ("directory", "zip"):
        raise ReleaseError("Format must be directory or zip")
    if destination.exists() or destination.is_symlink():
        raise ReleaseError("Destination already exists; choose a new path: %s" % destination)
    resolved_destination = destination.resolve()
    for directory in ALLOWED_DIRECTORIES:
        if resolved_destination.is_relative_to(source / directory):
            raise ReleaseError("Destination must be outside exported source directories")
    files = collect_sources(source)
    manifest = _manifest(files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "directory":
        # mkdir is exclusive; no pre-existing directory is reused or cleaned.
        destination.mkdir()
        try:
            for relative, (data, mode) in files.items():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(mode)
            (destination / MANIFEST_NAME).write_bytes(manifest)
        except BaseException:
            shutil.rmtree(destination)
            raise
    else:
        # Exclusive mode also rejects a destination created after the first check.
        with destination.open("xb") as output:
            try:
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for relative, (data, mode) in [*files.items(), (MANIFEST_NAME, (manifest, 0o644))]:
                        info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                        info.create_system = 3
                        info.external_attr = (stat.S_IFREG | mode) << 16
                        info.compress_type = zipfile.ZIP_DEFLATED
                        archive.writestr(info, data)
            except BaseException:
                output.close()
                destination.unlink()
                raise
    return {"destination": str(destination), "format": output_format,
            "source_files": len(files), "manifest": MANIFEST_NAME,
            "pretrained_binary_weights_included": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="New output path; existing paths are never overwritten")
    parser.add_argument("--format", choices=("directory", "zip"), default="directory", dest="output_format")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1], help="Source project root (default: this checkout)")
    options = parser.parse_args()
    try:
        result = prepare_release(options.source, options.destination, options.output_format)
    except (ReleaseError, OSError) as exc:
        parser.exit(2, "Release export failed: %s\n" % exc)
    print(json.dumps(result, indent=2))
    print("Excluded by default: pretrained binary weights, original PDF, local results/review/validation artifacts, environments, caches, and known secret files.")
    print("The source workspace was not modified. Review the export before publishing it to GitHub.")


if __name__ == "__main__":
    main()

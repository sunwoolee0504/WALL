"""Portable, checksum-verified release assets; no machine-specific paths."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src/config/release_catalog.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_path(root, relative):
    relative = Path(relative)
    root = Path(root).resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Release paths must be relative and contained in their root")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Release path escapes its root") from exc
    return candidate


def verified_asset(root, path, catalog):
    info = catalog["assets"][path]
    file = safe_path(root, path)
    if not file.is_file():
        raise FileNotFoundError(
            "Missing release asset: %s. Extract wall-final-20261030-models.zip at --asset-root."
            % file
        )
    if file.stat().st_size != info["bytes"] or digest(file) != info["sha256"]:
        raise ValueError("Release asset checksum mismatch: " + str(file))
    return file


def load_catalog(path=CATALOG):
    data = read_json(path)
    if data.get("format_version") != 1:
        raise ValueError("Unsupported release catalog version")
    return data


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)

#!/usr/bin/env bash
# Create a local virtual environment using requirements.txt.
# Wheel sources: https://docs.pytorch.org/get-started/previous-versions/
set -euo pipefail
WALL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: WALL_PYTHON_BIN=python3.10 bash scripts/install_optimized_env.sh [cpu|cu118|cu121|cu124] [new-venv-path]\nDefault: CPU wheel, project/.venv-wall. Python 3.10 or 3.11 is required.\n'
  exit 0
fi
if (( $# > 2 )); then
  printf 'Too many arguments; use --help.\n' >&2
  exit 2
fi
WALL_WHEEL="${1:-cpu}"
WALL_VENV="${2:-${WALL_PROJECT_ROOT}/.venv-wall}"
WALL_PYTHON_BIN="${WALL_PYTHON_BIN:-python3}"
case "$WALL_WHEEL" in
  cpu|cu118|cu121|cu124) ;;
  *) printf 'Wheel must be cpu, cu118, cu121, or cu124.\n' >&2; exit 2 ;;
esac
if [[ -e "$WALL_VENV" || -L "$WALL_VENV" ]]; then
  printf 'Refusing to modify an existing environment: %s\nChoose a new venv directory.\n' "$WALL_VENV" >&2
  exit 2
fi
"$WALL_PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 11), "Use WALL_PYTHON_BIN with Python 3.10 or 3.11"'
if [[ "$WALL_VENV" != /* ]]; then
  WALL_VENV="$PWD/$WALL_VENV"
fi
WALL_TORCH_REQUIREMENT="$("$WALL_PYTHON_BIN" -c 'from pathlib import Path; import sys; print(next(line for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith("torch==")))' "$WALL_PROJECT_ROOT/requirements.txt")"
"$WALL_PYTHON_BIN" -m venv "$WALL_VENV"
"$WALL_VENV/bin/python" -m pip install --upgrade pip
"$WALL_VENV/bin/python" -m pip install "$WALL_TORCH_REQUIREMENT" --index-url "https://download.pytorch.org/whl/${WALL_WHEEL}"
"$WALL_VENV/bin/python" -m pip install -r "$WALL_PROJECT_ROOT/requirements.txt"
"$WALL_VENV/bin/python" -m pip check
"$WALL_VENV/bin/python" -m pip freeze > "$WALL_VENV/requirements-resolved.txt"
"$WALL_VENV/bin/python" "$WALL_PROJECT_ROOT/scripts/check_environment.py"
printf '\nEnvironment created at %s\nActivate with: source %q/bin/activate\nSet SC2PATH or run bash scripts/install_sc2.sh before training.\n' "$WALL_VENV" "$WALL_VENV"

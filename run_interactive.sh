#!/usr/bin/env bash
set -euo pipefail
WALL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# == 0 ]]; then
  printf 'Usage: bash run_interactive.sh <cpu|GPU-index[,GPU-index]|all> [command ...]\nOpens bash by default; otherwise uses the same arguments and images as run.sh.\n'
  exit 0
fi
WALL_GPU="$1"
shift
if (( $# == 0 )); then set -- bash; fi
WALL_INTERACTIVE=1 exec bash "$WALL_PROJECT_ROOT/run.sh" "$WALL_GPU" "$@"

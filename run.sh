#!/usr/bin/env bash
# Container launcher; Python dependencies are installed only by the shared installer.
set -euo pipefail
WALL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# == 0 ]]; then
  printf 'Usage: bash run.sh <cpu|GPU-index[,GPU-index]|all> [command ...]\nExample: bash run.sh 0 python src/main.py --config=wall_qmix --env-config=sc2 with env_args.map_name=3m\nDefault images: wall-optimized:cpu / wall-optimized:cu121. Override with WALL_DOCKER_IMAGE.\nSet SC2PATH for an existing host installation. Default command only checks the environment.\n'
  exit 0
fi
WALL_GPU="$1"
shift
WALL_DOCKER_ARGS=(run --rm --init --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$WALL_PROJECT_ROOT,target=/workspace/wall" \
  --workdir /workspace/wall --env XDG_CACHE_HOME=/tmp/wall-cache)
if [[ "$WALL_GPU" == cpu ]]; then
  WALL_IMAGE="${WALL_DOCKER_IMAGE:-wall-optimized:cpu}"
elif [[ "$WALL_GPU" == all ]]; then
  WALL_IMAGE="${WALL_DOCKER_IMAGE:-wall-optimized:cu121}"
  WALL_DOCKER_ARGS+=(--gpus all)
elif [[ "$WALL_GPU" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  WALL_IMAGE="${WALL_DOCKER_IMAGE:-wall-optimized:cu121}"
  # Docker parses --gpus as CSV; keep a device list inside one quoted CSV field.
  WALL_DOCKER_ARGS+=(--gpus "\"device=$WALL_GPU\"")
else
  printf 'GPU selector must be cpu, all, or comma-separated device indices.\n' >&2
  exit 2
fi
if [[ "${WALL_INTERACTIVE:-0}" == 1 ]]; then
  WALL_DOCKER_ARGS+=(-i)
  if [[ -t 0 && -t 1 ]]; then WALL_DOCKER_ARGS+=(-t); fi
fi
WALL_SC2_ROOT="${SC2PATH:-}"
if [[ -z "$WALL_SC2_ROOT" && -d "$WALL_PROJECT_ROOT/3rdparty/StarCraftII" ]]; then
  WALL_SC2_ROOT="$WALL_PROJECT_ROOT/3rdparty/StarCraftII"
elif [[ -z "$WALL_SC2_ROOT" && -d "${HOME}/StarCraftII" ]]; then
  WALL_SC2_ROOT="${HOME}/StarCraftII"
fi
if [[ -n "$WALL_SC2_ROOT" ]]; then
  if [[ ! -d "$WALL_SC2_ROOT" ]]; then
    printf 'SC2PATH does not exist: %s\n' "$WALL_SC2_ROOT" >&2
    exit 2
  fi
  WALL_SC2_ROOT="$(cd -- "$WALL_SC2_ROOT" && pwd)"
  WALL_DOCKER_ARGS+=(--mount "type=bind,source=$WALL_SC2_ROOT,target=/opt/StarCraftII" --env SC2PATH=/opt/StarCraftII)
fi
if (( $# == 0 )); then set -- python scripts/check_environment.py; fi
exec docker "${WALL_DOCKER_ARGS[@]}" "$WALL_IMAGE" "$@"

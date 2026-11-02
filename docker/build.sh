#!/usr/bin/env bash
set -euo pipefail
WALL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: bash docker/build.sh [cpu|cu118|cu121|cu124] [image-tag]\nDefault image: wall-optimized:<wheel>. No SC2 binary or user data is copied into the image.\n'
  exit 0
fi
if (( $# > 2 )); then
  printf 'Too many arguments; use --help.\n' >&2
  exit 2
fi
WALL_WHEEL="${1:-cpu}"
case "$WALL_WHEEL" in cpu|cu118|cu121|cu124) ;; *) printf 'Unsupported wheel: %s\n' "$WALL_WHEEL" >&2; exit 2 ;; esac
WALL_IMAGE="${2:-wall-optimized:${WALL_WHEEL}}"
exec docker build --file "$WALL_PROJECT_ROOT/docker/Dockerfile" \
  --build-arg "WALL_WHEEL=$WALL_WHEEL" --tag "$WALL_IMAGE" "$WALL_PROJECT_ROOT"

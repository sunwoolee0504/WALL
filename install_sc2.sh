#!/usr/bin/env bash
# Explicit user-run SC2 4.10 + SMAC maps installer; Python setup is separate.
# Downloads: https://github.com/Blizzard/s2client-proto#downloads
# Maps: https://github.com/oxwhirl/smac#smac-maps
set -euo pipefail
WALL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: bash install_sc2.sh [StarCraftII-directory]\nUses SC2PATH when set, otherwise project/3rdparty/StarCraftII. Existing game/maps are preserved.\nThe Blizzard archive uses its published iagreetotheeula password; see the Blizzard download license.\n'
  exit 0
fi
if (( $# > 1 )); then printf 'Too many arguments; use --help.\n' >&2; exit 2; fi
WALL_SC2_ROOT="${1:-${SC2PATH:-${WALL_PROJECT_ROOT}/3rdparty/StarCraftII}}"
if [[ "$WALL_SC2_ROOT" != /* ]]; then WALL_SC2_ROOT="$PWD/$WALL_SC2_ROOT"; fi
WALL_TEMP_DIR=""
trap 'if [[ -n "$WALL_TEMP_DIR" ]]; then rm -rf -- "$WALL_TEMP_DIR"; fi' EXIT
wall_download() {
  command -v curl >/dev/null || { printf 'curl is required for downloads.\n' >&2; return 1; }
  command -v unzip >/dev/null || { printf 'unzip is required for installation.\n' >&2; return 1; }
  if [[ -z "$WALL_TEMP_DIR" ]]; then WALL_TEMP_DIR="$(mktemp -d)"; fi
  curl --fail --location --retry 3 --output "$2" "$1"
}
if [[ ! -d "$WALL_SC2_ROOT/Versions" ]]; then
  if [[ -e "$WALL_SC2_ROOT" ]]; then
    printf 'Existing directory is not a complete SC2 installation: %s\nChoose a new directory or finish that installation manually.\n' "$WALL_SC2_ROOT" >&2
    exit 2
  fi
  WALL_TEMP_DIR="$(mktemp -d)"
  wall_download https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip "$WALL_TEMP_DIR/SC2.zip"
  unzip -q -P iagreetotheeula "$WALL_TEMP_DIR/SC2.zip" -d "$WALL_TEMP_DIR"
  test -d "$WALL_TEMP_DIR/StarCraftII/Versions"
  mkdir -p -- "$(dirname -- "$WALL_SC2_ROOT")"
  mv -- "$WALL_TEMP_DIR/StarCraftII" "$WALL_SC2_ROOT"
fi
if [[ ! -d "$WALL_SC2_ROOT/Maps/SMAC_Maps" ]]; then
  if [[ -z "$WALL_TEMP_DIR" ]]; then WALL_TEMP_DIR="$(mktemp -d)"; fi
  wall_download https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip "$WALL_TEMP_DIR/SMAC_Maps.zip"
  unzip -q "$WALL_TEMP_DIR/SMAC_Maps.zip" -d "$WALL_TEMP_DIR/maps"
  test -d "$WALL_TEMP_DIR/maps/SMAC_Maps"
  mkdir -p -- "$WALL_SC2_ROOT/Maps"
  mv -- "$WALL_TEMP_DIR/maps/SMAC_Maps" "$WALL_SC2_ROOT/Maps/SMAC_Maps"
fi
printf 'SC2 and SMAC maps are ready. Set this in your shell:\nexport SC2PATH=%q\n' "$WALL_SC2_ROOT"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SHIFT_DIR="$ROOT_DIR"
VENV_PYTHON="$SHIFT_DIR/.venv/bin/python"
BASE_SITE_PACKAGES="/home/sergei/.virtualenvs/base/lib/python3.12/site-packages"
VENV_SITE_PACKAGES="$SHIFT_DIR/.venv/lib/python3.12/site-packages"
BASE_SITE_PTH="$VENV_SITE_PACKAGES/base_env.pth"
HF_CACHE_DIR="$SHIFT_DIR/.hf_cache"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    printf 'Missing required path: %s\n' "$path" >&2
    exit 1
  fi
}

prepare_local_python() {
  require_path "$SHIFT_DIR"
  require_path "$VENV_PYTHON"
  require_path "$BASE_SITE_PACKAGES"

  mkdir -p "$VENV_SITE_PACKAGES"
  printf '%s\n' "$BASE_SITE_PACKAGES" > "$BASE_SITE_PTH"
  mkdir -p "$HF_CACHE_DIR"
}

install_project_editable() {
  prepare_local_python
  (
    cd "$SHIFT_DIR"
    "$VENV_PYTHON" -m pip install -e . --no-build-isolation
  )
}

run_in_shift_python() {
  prepare_local_python
  (
    cd "$SHIFT_DIR"
    HF_HOME="$HF_CACHE_DIR" "$VENV_PYTHON" "$@"
  )
}

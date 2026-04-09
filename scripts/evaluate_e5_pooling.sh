#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_e5_common.sh"

install_project_editable
run_in_shift_python src/app/e5_evaluate.py \
  --dataset-dir data/e5_pooling \
  --checkpoint-path data/e5_pooling/checkpoints/best_pooling_checkpoint.pt \
  "$@"

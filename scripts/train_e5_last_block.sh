#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_e5_common.sh"

install_project_editable
run_in_shift_python src/app/e5_train.py \
  --dataset-dir data/e5_pooling \
  --output-dir data/e5_pooling/checkpoints_last_block \
  --train-last-transformer-block \
  "$@"

printf 'Training artifacts:\n'
printf '  %s\n' "$ROOT_DIR/data/e5_pooling/checkpoints_last_block/training_history.json"
printf '  %s\n' "$ROOT_DIR/data/e5_pooling/checkpoints_last_block/plots/training_curves.png"
printf '  %s\n' "$ROOT_DIR/data/e5_pooling/checkpoints_last_block/plots/training_curves.svg"

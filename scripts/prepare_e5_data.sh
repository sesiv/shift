#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_e5_common.sh"

install_project_editable
run_in_shift_python src/app/e5_prepare_data.py \
  --source src/data/ExportSDLab.xlsx \
  --output-dir data/e5_pooling \
  "$@"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"$SCRIPT_DIR/setup_e5_env.sh"
"$SCRIPT_DIR/prepare_e5_data.sh"
"$SCRIPT_DIR/train_e5_pooling.sh"
"$SCRIPT_DIR/evaluate_e5_pooling.sh"

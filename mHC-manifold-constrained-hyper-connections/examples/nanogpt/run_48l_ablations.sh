#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NPROC="${NPROC:-4}"
mkdir -p logs

echo "Using torchrun with ${NPROC} processes"

run() {
  local name="$1"
  shift
  echo "=== Running ${name} ==="
  torchrun --standalone --nproc_per_node="${NPROC}" train.py "$@" 2>&1 | tee "logs/${name}.log"
}

run baseline-48l config/train_fineweb10B_48l.py
run hc-48l config/train_fineweb10B_hc_48l.py
run mhc-48l config/train_fineweb10B_mhc_48l.py
run vres-48l config/train_fineweb10B_vres_48l.py
run mhc-ortho-48l config/train_fineweb10B_mhc_48l.py mhc_h_res_proj="orthostochastic" out_dir="out-fineweb10B-mhc-ortho-48l"

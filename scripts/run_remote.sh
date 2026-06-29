#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/run_remote.sh /path/to/matrix.mtx [dense_cols] [warmup] [repeat]"
  exit 1
fi

MATRIX_PATH="$1"
DENSE_COLS="${2:-128}"
WARMUP="${3:-5}"
REPEAT="${4:-20}"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/acc_spmm_bench --matrix "${MATRIX_PATH}" --n "${DENSE_COLS}" --warmup "${WARMUP}" --repeat "${REPEAT}" --check 1

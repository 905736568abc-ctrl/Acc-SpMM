#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/run_phase1_baseline.sh /path/to/datasets [warmup] [repeat]"
  exit 1
fi

DATASET_DIR="$1"
WARMUP="${2:-2}"
REPEAT="${3:-5}"
N_LIST=(32 64 128 256)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "matrix,rows,cols,nnz,density,n,warmup,repeat,kernel,average_ms,gflops,max_abs_diff"

if ! find "${DATASET_DIR}" -type f -name "*.mtx" | grep -q .; then
  echo "No .mtx files found under ${DATASET_DIR}" >&2
  exit 1
fi

find "${DATASET_DIR}" -type f -name "*.mtx" | sort | while read -r matrix; do
  for n in "${N_LIST[@]}"; do
    ./build/acc_spmm_bench --matrix "${matrix}" --n "${n}" --warmup "${WARMUP}" --repeat "${REPEAT}" --check 1 \
      | grep '^csv,' \
      | sed 's/^csv,//' \
      | sed 's/matrix=//; s/,rows=/,/; s/,cols=/,/; s/,nnz=/,/; s/,density=/,/; s/,n=/,/; s/,warmup=/,/; s/,repeat=/,/; s/,kernel=/,/; s/,average_ms=/,/; s/,gflops=/,/; s/,max_abs_diff=/,/'
  done
done

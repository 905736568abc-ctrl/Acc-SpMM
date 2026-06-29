# Acc-SpMM-Reproduce

This repository is a remote-first reproduction scaffold for the paper "Acc-SpMM: Accelerating General-purpose Sparse Matrix-Matrix Multiplication with GPU Tensor Cores".

## Repository Layout

- `acc/`: reordering, BitTCF, and pipeline-related interfaces
- `baseline/`: cuSPARSE baseline implementation
- `benchmark/`: Matrix Market loader and benchmark entry
- `datasets/`: local dataset placement
- `docs/papers/`: reference paper PDF
- `dtc/`: DTC-SpMM placeholder baseline
- `scripts/`: remote build and run helper scripts

## Current Scope

- Matrix Market loader for SuiteSparse-style `.mtx` files
- cuSPARSE baseline benchmark
- CPU reference checker for correctness
- Placeholders for DTC, reordering, BitTCF, and pipeline modules

## Quick Start On A Remote Linux Server

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/acc_spmm_bench --matrix ./datasets/example.mtx --n 128 --warmup 5 --repeat 20 --check 1
```

## Expected Inputs

- Sparse matrix `A` in Matrix Market format
- Dense matrix `B` is generated deterministically inside the benchmark
- Output matrix `C` is validated against a CPU reference when `--check 1`

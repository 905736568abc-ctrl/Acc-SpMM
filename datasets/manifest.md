# Dataset Manifest

This file tracks the datasets currently prepared for `Acc-SpMM-Reproduce`.

## Smoke

- `smoke/test.mtx`
  - Type: smoke test
  - Format: Matrix Market `coordinate real general`
  - Purpose: correctness check, loader validation, and benchmark smoke test
  - Status: ready

## SuiteSparse

- `suitesparse/bcsstk17/bcsstk17.mtx`
  - Source group: SuiteSparse
  - Format: Matrix Market `coordinate real symmetric`
  - Purpose: structured sparse matrix for baseline evaluation
  - Status: ready

- `suitesparse/west0067.tar.gz`
  - Source group: SuiteSparse
  - Purpose: archived dataset package
  - Status: downloaded, `.mtx` not extracted yet

## Graph

- `graph/amazon0312/amazon0312.mtx`
  - Source group: graph / power-law style
  - Purpose: graph-style sparse matrix for baseline and later reordering experiments
  - Status: ready

- `graph/webbase-1M/webbase-1M.mtx`
  - Source group: graph / web graph
  - Purpose: larger graph-style sparse matrix for baseline and later load-balance experiments
  - Status: ready

## Archived Packages

- `graph/amazon0312.tar.gz`
- `graph/webbase-1M.tar.gz`
- `suitesparse/bcsstk17.tar.gz`
- `suitesparse/west0067.tar.gz`

These archives are kept for reproducibility, while benchmark runs should use the extracted `.mtx` files.

#pragma once

#include "acc/bittcf.hpp"

namespace acc_spmm {

KernelResult run_dtc_placeholder(const CsrMatrix& matrix, const DenseMatrix& rhs, int warmup, int repeat);

}  // namespace acc_spmm

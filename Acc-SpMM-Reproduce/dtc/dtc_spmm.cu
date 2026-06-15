#include "dtc/dtc_spmm.hpp"

#include <stdexcept>

namespace acc_spmm {

KernelResult run_dtc_placeholder(const CsrMatrix& matrix, const DenseMatrix& rhs, int warmup, int repeat) {
    (void)matrix;
    (void)rhs;
    (void)warmup;
    (void)repeat;
    throw std::runtime_error("DTC-SpMM baseline is not implemented yet.");
}

}  // namespace acc_spmm

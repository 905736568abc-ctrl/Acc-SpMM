#pragma once

#include "acc/bittcf.hpp"

namespace acc_spmm {

PipelineConfig make_default_pipeline();
void launch_pipeline_spmm(const CsrMatrix& matrix, const DenseMatrix& rhs, DenseMatrix* out, const PipelineConfig& config);
KernelResult run_pipeline_placeholder(const CsrMatrix& matrix,
                                      const DenseMatrix& rhs,
                                      const BittcfFormat& format,
                                      int warmup,
                                      int repeat,
                                      const PipelineConfig& config);

}  // namespace acc_spmm

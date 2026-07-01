#include "acc/pipeline.hpp"
#include "baseline/cusparse_spmm.hpp"

#include <stdexcept>

namespace acc_spmm {

PipelineConfig make_default_pipeline() {
    return {};
}

void launch_pipeline_spmm(const CsrMatrix& matrix, const DenseMatrix& rhs, DenseMatrix* out, const PipelineConfig& config) {
    (void)matrix;
    (void)rhs;
    (void)out;
    (void)config;
}

KernelResult run_pipeline_placeholder(const CsrMatrix& matrix,
                                      const DenseMatrix& rhs,
                                      const BittcfFormat& format,
                                      int warmup,
                                      int repeat,
                                      const PipelineConfig& config) {
    if (format.row_window_count <= 0 || format.tc_block_count <= 0) {
        throw std::runtime_error("Pipeline placeholder requires a non-empty BitTCF format.");
    }
    (void)config;
    return run_cusparse_spmm(matrix, rhs, warmup, repeat);
}

}  // namespace acc_spmm

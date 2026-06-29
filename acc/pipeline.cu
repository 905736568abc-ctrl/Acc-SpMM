#include "acc/pipeline.hpp"

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

}  // namespace acc_spmm

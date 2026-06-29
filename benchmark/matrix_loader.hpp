#pragma once

#include "acc/bittcf.hpp"

#include <string>

namespace acc_spmm {

CsrMatrix load_matrix_market(const std::string& path);
DenseMatrix make_dense_matrix(int rows, int cols, float seed = 0.001f);
DenseMatrix compute_reference_spmm(const CsrMatrix& matrix, const DenseMatrix& rhs);
float max_abs_diff(const DenseMatrix& lhs, const DenseMatrix& rhs);

}  // namespace acc_spmm

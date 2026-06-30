#pragma once

#include "acc/bittcf.hpp"

#include <string>

namespace acc_spmm {

CsrMatrix load_matrix_market(const std::string& path);
DenseMatrix make_dense_matrix(int rows, int cols, float seed = 0.001f);
DenseMatrix compute_reference_spmm(const CsrMatrix& matrix, const DenseMatrix& rhs);
float max_abs_value(const DenseMatrix& matrix);
float max_abs_diff(const DenseMatrix& lhs, const DenseMatrix& rhs);
float max_relative_diff(const DenseMatrix& reference, const DenseMatrix& candidate);
double matrix_density(const CsrMatrix& matrix);
double spmm_flop_count(const CsrMatrix& matrix, int dense_cols);

}  // namespace acc_spmm

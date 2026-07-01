#pragma once

#include "acc/bittcf.hpp"

namespace acc_spmm {

ReorderPlan build_affinity_reorder(const CsrMatrix& matrix, int tile_cols = 16);
CsrMatrix apply_reorder(const CsrMatrix& matrix, const ReorderPlan& plan);
DenseMatrix apply_rhs_reorder(const DenseMatrix& rhs, const ReorderPlan& plan);
DenseMatrix restore_output_row_order(const DenseMatrix& reordered_output, const ReorderPlan& plan);
ReorderDecision evaluate_reorder_decision(const ReorderPlan& plan,
                                          const BittcfFormat& original_format,
                                          const BittcfFormat& reordered_format);

}  // namespace acc_spmm

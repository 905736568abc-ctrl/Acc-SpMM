#pragma once

#include "acc/bittcf.hpp"

namespace acc_spmm {

ReorderPlan build_affinity_reorder(const CsrMatrix& matrix, int tile_cols = 16);
CsrMatrix apply_reorder(const CsrMatrix& matrix, const ReorderPlan& plan);

}  // namespace acc_spmm

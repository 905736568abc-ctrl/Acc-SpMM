#pragma once

#include "acc/bittcf.hpp"

namespace acc_spmm {

ReorderPlan build_affinity_reorder(const CsrMatrix& matrix);

}  // namespace acc_spmm

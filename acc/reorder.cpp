#include "acc/reorder.hpp"

#include <numeric>

namespace acc_spmm {

ReorderPlan build_affinity_reorder(const CsrMatrix& matrix) {
    ReorderPlan plan;
    plan.permutation.resize(matrix.rows);
    std::iota(plan.permutation.begin(), plan.permutation.end(), 0);
    return plan;
}

}  // namespace acc_spmm

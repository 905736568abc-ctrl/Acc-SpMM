#include "acc/reorder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace acc_spmm {
namespace {

struct RowFeature {
    int row = 0;
    int nnz = 0;
    int first_tile = -1;
    int span = 0;
    std::vector<int> tiles;
};

RowFeature analyze_row(const CsrMatrix& matrix, int row, int tile_cols) {
    RowFeature feature;
    feature.row = row;

    const int begin = matrix.row_offsets[static_cast<size_t>(row)];
    const int end = matrix.row_offsets[static_cast<size_t>(row) + 1];
    feature.nnz = end - begin;
    if (feature.nnz == 0) {
        return feature;
    }

    int min_col = matrix.cols;
    int max_col = -1;
    feature.tiles.reserve(static_cast<size_t>(feature.nnz));
    for (int idx = begin; idx < end; ++idx) {
        const int col = matrix.col_indices[static_cast<size_t>(idx)];
        min_col = std::min(min_col, col);
        max_col = std::max(max_col, col);
        feature.tiles.push_back(col / tile_cols);
    }

    std::sort(feature.tiles.begin(), feature.tiles.end());
    feature.tiles.erase(std::unique(feature.tiles.begin(), feature.tiles.end()), feature.tiles.end());
    feature.first_tile = feature.tiles.front();
    feature.span = max_col - min_col + 1;
    return feature;
}

double tile_jaccard(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    if (lhs.empty() && rhs.empty()) {
        return 1.0;
    }

    size_t i = 0;
    size_t j = 0;
    int intersection = 0;
    int union_count = 0;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i] == rhs[j]) {
            ++intersection;
            ++union_count;
            ++i;
            ++j;
        } else if (lhs[i] < rhs[j]) {
            ++union_count;
            ++i;
        } else {
            ++union_count;
            ++j;
        }
    }
    union_count += static_cast<int>(lhs.size() - i + rhs.size() - j);
    return union_count > 0 ? static_cast<double>(intersection) / static_cast<double>(union_count) : 0.0;
}

ReorderMetrics compute_metrics_from_features(const std::vector<RowFeature>& features, int tile_cols) {
    ReorderMetrics metrics;
    double row_nnz_sum = 0.0;
    double row_span_sum = 0.0;
    double tile_count_sum = 0.0;
    double tile_occupancy_sum = 0.0;
    double jaccard_sum = 0.0;
    double tile_distance_sum = 0.0;

    std::vector<RowFeature> nonempty_rows;
    nonempty_rows.reserve(features.size());
    for (const RowFeature& feature : features) {
        if (feature.nnz == 0) {
            continue;
        }
        metrics.nonempty_rows += 1;
        metrics.max_row_nnz = std::max(metrics.max_row_nnz, feature.nnz);
        row_nnz_sum += feature.nnz;
        row_span_sum += feature.span;
        tile_count_sum += static_cast<double>(feature.tiles.size());
        tile_occupancy_sum += static_cast<double>(feature.nnz) /
                              static_cast<double>(std::max<int>(1, static_cast<int>(feature.tiles.size()) * tile_cols));
        nonempty_rows.push_back(feature);
    }

    for (size_t idx = 1; idx < nonempty_rows.size(); ++idx) {
        jaccard_sum += tile_jaccard(nonempty_rows[idx - 1].tiles, nonempty_rows[idx].tiles);
        tile_distance_sum += std::fabs(static_cast<double>(nonempty_rows[idx].first_tile - nonempty_rows[idx - 1].first_tile));
    }

    if (metrics.nonempty_rows > 0) {
        const double count = static_cast<double>(metrics.nonempty_rows);
        metrics.avg_row_nnz = row_nnz_sum / count;
        metrics.avg_row_span = row_span_sum / count;
        metrics.avg_unique_col_tiles = tile_count_sum / count;
        metrics.avg_tile_occupancy = tile_occupancy_sum / count;
    }
    if (nonempty_rows.size() > 1) {
        const double adjacent_count = static_cast<double>(nonempty_rows.size() - 1);
        metrics.adjacent_tile_jaccard = jaccard_sum / adjacent_count;
        metrics.adjacent_first_tile_distance = tile_distance_sum / adjacent_count;
    }
    return metrics;
}

std::vector<RowFeature> build_feature_sequence(const CsrMatrix& matrix, const std::vector<int>& order, int tile_cols) {
    std::vector<RowFeature> features;
    features.reserve(order.size());
    for (int row : order) {
        features.push_back(analyze_row(matrix, row, tile_cols));
    }
    return features;
}

}  // namespace

ReorderPlan build_affinity_reorder(const CsrMatrix& matrix, int tile_cols) {
    ReorderPlan plan;
    if (tile_cols <= 0) {
        throw std::runtime_error("tile_cols must be positive.");
    }

    plan.permutation.resize(matrix.rows);
    std::iota(plan.permutation.begin(), plan.permutation.end(), 0);

    const std::vector<RowFeature> original_features = build_feature_sequence(matrix, plan.permutation, tile_cols);
    plan.original_metrics = compute_metrics_from_features(original_features, tile_cols);

    std::stable_sort(plan.permutation.begin(), plan.permutation.end(), [&](int lhs_row, int rhs_row) {
        const RowFeature& lhs = original_features[static_cast<size_t>(lhs_row)];
        const RowFeature& rhs = original_features[static_cast<size_t>(rhs_row)];

        const bool lhs_empty = lhs.nnz == 0;
        const bool rhs_empty = rhs.nnz == 0;
        if (lhs_empty != rhs_empty) {
            return !lhs_empty;
        }
        if (lhs.first_tile != rhs.first_tile) {
            return lhs.first_tile < rhs.first_tile;
        }
        if (lhs.tiles.size() != rhs.tiles.size()) {
            return lhs.tiles.size() < rhs.tiles.size();
        }
        if (lhs.span != rhs.span) {
            return lhs.span < rhs.span;
        }
        if (lhs.nnz != rhs.nnz) {
            return lhs.nnz > rhs.nnz;
        }
        return lhs.row < rhs.row;
    });

    const std::vector<RowFeature> reordered_features = build_feature_sequence(matrix, plan.permutation, tile_cols);
    plan.reordered_metrics = compute_metrics_from_features(reordered_features, tile_cols);
    return plan;
}

CsrMatrix apply_reorder(const CsrMatrix& matrix, const ReorderPlan& plan) {
    if (static_cast<int>(plan.permutation.size()) != matrix.rows) {
        throw std::runtime_error("Reorder permutation size does not match matrix rows.");
    }

    CsrMatrix reordered;
    reordered.rows = matrix.rows;
    reordered.cols = matrix.cols;
    reordered.row_offsets.assign(static_cast<size_t>(matrix.rows) + 1, 0);
    reordered.col_indices.reserve(matrix.col_indices.size());
    reordered.values.reserve(matrix.values.size());

    for (int new_row = 0; new_row < matrix.rows; ++new_row) {
        const int old_row = plan.permutation[static_cast<size_t>(new_row)];
        const int begin = matrix.row_offsets[static_cast<size_t>(old_row)];
        const int end = matrix.row_offsets[static_cast<size_t>(old_row) + 1];
        reordered.row_offsets[static_cast<size_t>(new_row) + 1] =
            reordered.row_offsets[static_cast<size_t>(new_row)] + (end - begin);
        for (int idx = begin; idx < end; ++idx) {
            reordered.col_indices.push_back(matrix.col_indices[static_cast<size_t>(idx)]);
            reordered.values.push_back(matrix.values[static_cast<size_t>(idx)]);
        }
    }

    return reordered;
}

}  // namespace acc_spmm

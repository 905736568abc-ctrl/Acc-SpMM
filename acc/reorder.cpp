#include "acc/reorder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

struct Community {
    std::vector<int> rows;
    std::vector<int> union_tiles;
    int first_tile = -1;
    int total_nnz = 0;
    int max_span = 0;
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

int tile_intersection_count(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    size_t i = 0;
    size_t j = 0;
    int intersection = 0;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i] == rhs[j]) {
            ++intersection;
            ++i;
            ++j;
        } else if (lhs[i] < rhs[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return intersection;
}

std::vector<int> union_sorted_tiles(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    std::vector<int> result;
    result.reserve(lhs.size() + rhs.size());
    size_t i = 0;
    size_t j = 0;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i] == rhs[j]) {
            result.push_back(lhs[i]);
            ++i;
            ++j;
        } else if (lhs[i] < rhs[j]) {
            result.push_back(lhs[i++]);
        } else {
            result.push_back(rhs[j++]);
        }
    }
    while (i < lhs.size()) {
        result.push_back(lhs[i++]);
    }
    while (j < rhs.size()) {
        result.push_back(rhs[j++]);
    }
    return result;
}

double modularity_like_gain(const RowFeature& lhs, const RowFeature& rhs, double total_tile_refs) {
    const int shared = tile_intersection_count(lhs.tiles, rhs.tiles);
    if (shared == 0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double expected = (static_cast<double>(lhs.tiles.size()) * static_cast<double>(rhs.tiles.size())) /
                            std::max(1.0, total_tile_refs);
    const double locality_bonus = 1.0 /
                                  (1.0 + std::fabs(static_cast<double>(lhs.first_tile - rhs.first_tile)));
    return static_cast<double>(shared) - expected + 0.05 * locality_bonus;
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

std::vector<int> build_degree_order(const std::vector<RowFeature>& features) {
    std::vector<int> order(features.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs_row, int rhs_row) {
        const RowFeature& lhs = features[static_cast<size_t>(lhs_row)];
        const RowFeature& rhs = features[static_cast<size_t>(rhs_row)];
        const bool lhs_empty = lhs.nnz == 0;
        const bool rhs_empty = rhs.nnz == 0;
        if (lhs_empty != rhs_empty) {
            return !lhs_empty;
        }
        if (lhs.tiles.size() != rhs.tiles.size()) {
            return lhs.tiles.size() < rhs.tiles.size();
        }
        if (lhs.nnz != rhs.nnz) {
            return lhs.nnz < rhs.nnz;
        }
        if (lhs.first_tile != rhs.first_tile) {
            return lhs.first_tile < rhs.first_tile;
        }
        return lhs.row < rhs.row;
    });
    return order;
}

std::vector<std::vector<int>> build_tile_to_rows(const std::vector<RowFeature>& features,
                                                 const std::vector<int>& degree_order,
                                                 int tile_cols,
                                                 int matrix_cols) {
    const int tile_count = std::max(1, (matrix_cols + tile_cols - 1) / tile_cols);
    std::vector<std::vector<int>> tile_to_rows(static_cast<size_t>(tile_count));
    for (int row : degree_order) {
        for (int tile : features[static_cast<size_t>(row)].tiles) {
            tile_to_rows[static_cast<size_t>(tile)].push_back(row);
        }
    }
    return tile_to_rows;
}

Community make_singleton_community(const RowFeature& feature) {
    Community community;
    community.rows.push_back(feature.row);
    community.union_tiles = feature.tiles;
    community.first_tile = feature.first_tile;
    community.total_nnz = feature.nnz;
    community.max_span = feature.span;
    return community;
}

Community make_pair_community(const RowFeature& lhs, const RowFeature& rhs) {
    Community community;
    community.rows = {lhs.row, rhs.row};
    std::stable_sort(community.rows.begin(), community.rows.end(), [&](int lhs_row, int rhs_row) {
        const RowFeature& l = lhs_row == lhs.row ? lhs : rhs;
        const RowFeature& r = rhs_row == lhs.row ? lhs : rhs;
        if (l.tiles != r.tiles) {
            return l.tiles < r.tiles;
        }
        if (l.span != r.span) {
            return l.span < r.span;
        }
        if (l.nnz != r.nnz) {
            return l.nnz > r.nnz;
        }
        return l.row < r.row;
    });
    community.union_tiles = union_sorted_tiles(lhs.tiles, rhs.tiles);
    community.first_tile = community.union_tiles.empty() ? -1 : community.union_tiles.front();
    community.total_nnz = lhs.nnz + rhs.nnz;
    community.max_span = std::max(lhs.span, rhs.span);
    return community;
}

std::vector<Community> build_modularity_like_communities(const std::vector<RowFeature>& features,
                                                         int tile_cols,
                                                         int matrix_cols) {
    constexpr int kMaxCandidatesPerTile = 32;

    const std::vector<int> degree_order = build_degree_order(features);
    const std::vector<std::vector<int>> tile_to_rows = build_tile_to_rows(features, degree_order, tile_cols, matrix_cols);
    const double total_tile_refs =
        std::max(1.0, std::accumulate(features.begin(),
                                      features.end(),
                                      0.0,
                                      [](double acc, const RowFeature& feature) {
                                          return acc + static_cast<double>(feature.tiles.size());
                                      }));

    std::vector<char> assigned(features.size(), 0);
    std::vector<int> visit_stamp(features.size(), -1);
    int current_stamp = 0;
    std::vector<Community> communities;
    communities.reserve(features.size());

    for (int row : degree_order) {
        if (assigned[static_cast<size_t>(row)]) {
            continue;
        }

        const RowFeature& source = features[static_cast<size_t>(row)];
        assigned[static_cast<size_t>(row)] = 1;
        if (source.nnz == 0 || source.tiles.empty()) {
            communities.push_back(make_singleton_community(source));
            continue;
        }

        int best_row = -1;
        int best_overlap = -1;
        double best_gain = 0.0;
        ++current_stamp;

        for (int tile : source.tiles) {
            int examined = 0;
            for (int candidate : tile_to_rows[static_cast<size_t>(tile)]) {
                if (examined >= kMaxCandidatesPerTile) {
                    break;
                }
                if (candidate == row || assigned[static_cast<size_t>(candidate)]) {
                    continue;
                }
                if (visit_stamp[static_cast<size_t>(candidate)] == current_stamp) {
                    continue;
                }
                visit_stamp[static_cast<size_t>(candidate)] = current_stamp;
                examined += 1;

                const RowFeature& neighbor = features[static_cast<size_t>(candidate)];
                const int overlap = tile_intersection_count(source.tiles, neighbor.tiles);
                const double gain = modularity_like_gain(source, neighbor, total_tile_refs);
                if (gain > best_gain ||
                    (std::fabs(gain - best_gain) < 1.0e-12 && overlap > best_overlap) ||
                    (std::fabs(gain - best_gain) < 1.0e-12 && overlap == best_overlap &&
                     neighbor.tiles.size() < features[static_cast<size_t>(best_row >= 0 ? best_row : candidate)].tiles.size()) ||
                    (std::fabs(gain - best_gain) < 1.0e-12 && overlap == best_overlap &&
                     neighbor.tiles.size() == features[static_cast<size_t>(best_row >= 0 ? best_row : candidate)].tiles.size() &&
                     candidate < (best_row >= 0 ? best_row : candidate))) {
                    best_gain = gain;
                    best_overlap = overlap;
                    best_row = candidate;
                }
            }
        }

        if (best_row >= 0 && best_gain > 0.0) {
            assigned[static_cast<size_t>(best_row)] = 1;
            communities.push_back(make_pair_community(source, features[static_cast<size_t>(best_row)]));
        } else {
            communities.push_back(make_singleton_community(source));
        }
    }

    return communities;
}

std::vector<int> flatten_communities(std::vector<Community> communities) {
    std::stable_sort(communities.begin(), communities.end(), [](const Community& lhs, const Community& rhs) {
        if (lhs.union_tiles.size() != rhs.union_tiles.size()) {
            return lhs.union_tiles.size() < rhs.union_tiles.size();
        }
        if (lhs.union_tiles != rhs.union_tiles) {
            return lhs.union_tiles < rhs.union_tiles;
        }
        if (lhs.max_span != rhs.max_span) {
            return lhs.max_span < rhs.max_span;
        }
        if (lhs.total_nnz != rhs.total_nnz) {
            return lhs.total_nnz > rhs.total_nnz;
        }
        return lhs.rows < rhs.rows;
    });

    std::vector<int> order;
    for (const Community& community : communities) {
        order.insert(order.end(), community.rows.begin(), community.rows.end());
    }
    return order;
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
    plan.permutation = flatten_communities(build_modularity_like_communities(original_features, tile_cols, matrix.cols));

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

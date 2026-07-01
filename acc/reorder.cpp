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

struct GraphData {
    int vertex_count = 0;
    int edge_count = 0;
    std::vector<std::vector<int>> neighbors;
    std::vector<int> degrees;
};

struct TreeNode {
    int left = -1;
    int right = -1;
    int leaf_vertex = -1;
};

struct DsuState {
    std::vector<int> parent;
    std::vector<int> size;
    std::vector<int> community_degree_sum;
    std::vector<int> tree_root;
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

int common_neighbor_count(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    size_t i = 0;
    size_t j = 0;
    int common = 0;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i] == rhs[j]) {
            ++common;
            ++i;
            ++j;
        } else if (lhs[i] < rhs[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return common;
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
        tile_distance_sum +=
            std::fabs(static_cast<double>(nonempty_rows[idx].first_tile - nonempty_rows[idx - 1].first_tile));
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

std::vector<RowFeature> build_feature_sequence(const CsrMatrix& matrix, int tile_cols) {
    std::vector<RowFeature> features;
    features.reserve(static_cast<size_t>(matrix.rows));
    for (int row = 0; row < matrix.rows; ++row) {
        features.push_back(analyze_row(matrix, row, tile_cols));
    }
    return features;
}

GraphData build_graph_from_matrix(const CsrMatrix& matrix) {
    GraphData graph;
    graph.vertex_count = std::max(matrix.rows, matrix.cols);
    graph.neighbors.assign(static_cast<size_t>(graph.vertex_count), {});

    for (int row = 0; row < matrix.rows; ++row) {
        const int begin = matrix.row_offsets[static_cast<size_t>(row)];
        const int end = matrix.row_offsets[static_cast<size_t>(row) + 1];
        for (int idx = begin; idx < end; ++idx) {
            const int col = matrix.col_indices[static_cast<size_t>(idx)];
            if (row == col) {
                graph.neighbors[static_cast<size_t>(row)].push_back(col);
            } else {
                graph.neighbors[static_cast<size_t>(row)].push_back(col);
                graph.neighbors[static_cast<size_t>(col)].push_back(row);
            }
        }
    }

    graph.degrees.assign(static_cast<size_t>(graph.vertex_count), 0);
    int total_adjacency = 0;
    int self_loops = 0;
    for (int vertex = 0; vertex < graph.vertex_count; ++vertex) {
        auto& nbrs = graph.neighbors[static_cast<size_t>(vertex)];
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        graph.degrees[static_cast<size_t>(vertex)] = static_cast<int>(nbrs.size());
        total_adjacency += static_cast<int>(nbrs.size());
        if (std::binary_search(nbrs.begin(), nbrs.end(), vertex)) {
            self_loops += 1;
        }
    }

    graph.edge_count = (total_adjacency - self_loops) / 2 + self_loops;
    return graph;
}

DsuState make_dsu(const GraphData& graph, std::vector<TreeNode>* tree_nodes) {
    DsuState dsu;
    const size_t n = static_cast<size_t>(graph.vertex_count);
    dsu.parent.resize(n);
    dsu.size.assign(n, 1);
    dsu.community_degree_sum = graph.degrees;
    dsu.tree_root.resize(n);
    std::iota(dsu.parent.begin(), dsu.parent.end(), 0);
    tree_nodes->reserve(n * 2);
    for (int vertex = 0; vertex < graph.vertex_count; ++vertex) {
        tree_nodes->push_back(TreeNode{.left = -1, .right = -1, .leaf_vertex = vertex});
        dsu.tree_root[static_cast<size_t>(vertex)] = vertex;
    }
    return dsu;
}

int dsu_find(DsuState* dsu, int node) {
    int root = node;
    while (dsu->parent[static_cast<size_t>(root)] != root) {
        root = dsu->parent[static_cast<size_t>(root)];
    }
    while (dsu->parent[static_cast<size_t>(node)] != node) {
        const int parent = dsu->parent[static_cast<size_t>(node)];
        dsu->parent[static_cast<size_t>(node)] = root;
        node = parent;
    }
    return root;
}

int dsu_union(DsuState* dsu, int lhs, int rhs, std::vector<TreeNode>* tree_nodes) {
    int lhs_root = dsu_find(dsu, lhs);
    int rhs_root = dsu_find(dsu, rhs);
    if (lhs_root == rhs_root) {
        return lhs_root;
    }

    if (dsu->size[static_cast<size_t>(lhs_root)] < dsu->size[static_cast<size_t>(rhs_root)]) {
        std::swap(lhs_root, rhs_root);
    }

    const int new_tree_root = static_cast<int>(tree_nodes->size());
    tree_nodes->push_back(TreeNode{
        .left = dsu->tree_root[static_cast<size_t>(lhs_root)],
        .right = dsu->tree_root[static_cast<size_t>(rhs_root)],
        .leaf_vertex = -1,
    });

    dsu->parent[static_cast<size_t>(rhs_root)] = lhs_root;
    dsu->size[static_cast<size_t>(lhs_root)] += dsu->size[static_cast<size_t>(rhs_root)];
    dsu->community_degree_sum[static_cast<size_t>(lhs_root)] +=
        dsu->community_degree_sum[static_cast<size_t>(rhs_root)];
    dsu->tree_root[static_cast<size_t>(lhs_root)] = new_tree_root;
    return lhs_root;
}

std::vector<int> build_degree_order(const GraphData& graph) {
    std::vector<int> order(static_cast<size_t>(graph.vertex_count));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        if (graph.degrees[static_cast<size_t>(lhs)] != graph.degrees[static_cast<size_t>(rhs)]) {
            return graph.degrees[static_cast<size_t>(lhs)] < graph.degrees[static_cast<size_t>(rhs)];
        }
        return lhs < rhs;
    });
    return order;
}

std::vector<int> build_tree_roots(const GraphData& graph, DsuState* dsu) {
    std::vector<int> roots;
    roots.reserve(static_cast<size_t>(graph.vertex_count));
    std::vector<char> seen(static_cast<size_t>(graph.vertex_count), 0);
    for (int vertex = 0; vertex < graph.vertex_count; ++vertex) {
        const int root = dsu_find(dsu, vertex);
        if (!seen[static_cast<size_t>(root)]) {
            seen[static_cast<size_t>(root)] = 1;
            roots.push_back(dsu->tree_root[static_cast<size_t>(root)]);
        }
    }
    return roots;
}

void collect_leaves_dfs(const std::vector<TreeNode>& tree_nodes, int node, std::vector<int>* leaves) {
    if (node < 0) {
        return;
    }
    const TreeNode& tree_node = tree_nodes[static_cast<size_t>(node)];
    if (tree_node.leaf_vertex >= 0) {
        leaves->push_back(tree_node.leaf_vertex);
        return;
    }
    collect_leaves_dfs(tree_nodes, tree_node.left, leaves);
    collect_leaves_dfs(tree_nodes, tree_node.right, leaves);
}

std::vector<int> order_subtree_vertices(const GraphData& graph,
                                        const std::vector<int>& dfs_leaves,
                                        const std::vector<int>& component_of_vertex) {
    if (dfs_leaves.empty()) {
        return {};
    }

    std::vector<char> in_subtree(static_cast<size_t>(graph.vertex_count), 0);
    for (int vertex : dfs_leaves) {
        in_subtree[static_cast<size_t>(vertex)] = 1;
    }

    std::vector<char> visited(static_cast<size_t>(graph.vertex_count), 0);
    std::vector<int> ordered;
    ordered.reserve(dfs_leaves.size());

    int current = dfs_leaves.front();
    visited[static_cast<size_t>(current)] = 1;
    ordered.push_back(current);

    while (ordered.size() < dfs_leaves.size()) {
        int best_next = -1;
        int best_common = -1;
        for (int candidate : graph.neighbors[static_cast<size_t>(current)]) {
            if (!in_subtree[static_cast<size_t>(candidate)] || visited[static_cast<size_t>(candidate)]) {
                continue;
            }
            if (component_of_vertex[static_cast<size_t>(candidate)] != component_of_vertex[static_cast<size_t>(current)]) {
                continue;
            }
            const int common = common_neighbor_count(graph.neighbors[static_cast<size_t>(current)],
                                                     graph.neighbors[static_cast<size_t>(candidate)]);
            if (common > best_common ||
                (common == best_common &&
                 graph.degrees[static_cast<size_t>(candidate)] < graph.degrees[static_cast<size_t>(best_next >= 0 ? best_next : candidate)]) ||
                (common == best_common &&
                 graph.degrees[static_cast<size_t>(candidate)] ==
                     graph.degrees[static_cast<size_t>(best_next >= 0 ? best_next : candidate)] &&
                 candidate < (best_next >= 0 ? best_next : candidate))) {
                best_common = common;
                best_next = candidate;
            }
        }

        if (best_next < 0) {
            for (int vertex : dfs_leaves) {
                if (!visited[static_cast<size_t>(vertex)]) {
                    best_next = vertex;
                    break;
                }
            }
        }

        visited[static_cast<size_t>(best_next)] = 1;
        ordered.push_back(best_next);
        current = best_next;
    }

    return ordered;
}

std::vector<int> build_vertex_order(const GraphData& graph,
                                    const std::vector<TreeNode>& tree_nodes,
                                    DsuState* dsu) {
    const std::vector<int> roots = build_tree_roots(graph, dsu);
    std::vector<int> component_of_vertex(static_cast<size_t>(graph.vertex_count), -1);
    for (int vertex = 0; vertex < graph.vertex_count; ++vertex) {
        component_of_vertex[static_cast<size_t>(vertex)] = dsu_find(dsu, vertex);
    }

    struct RootLeaves {
        int root = -1;
        std::vector<int> leaves;
    };

    std::vector<RootLeaves> root_leaves;
    root_leaves.reserve(roots.size());
    for (int root : roots) {
        RootLeaves item;
        item.root = root;
        collect_leaves_dfs(tree_nodes, root, &item.leaves);
        root_leaves.push_back(std::move(item));
    }

    std::stable_sort(root_leaves.begin(), root_leaves.end(), [&](const RootLeaves& lhs, const RootLeaves& rhs) {
        const int lhs_first = lhs.leaves.empty() ? graph.vertex_count : lhs.leaves.front();
        const int rhs_first = rhs.leaves.empty() ? graph.vertex_count : rhs.leaves.front();
        if (lhs.leaves.empty() != rhs.leaves.empty()) {
            return !lhs.leaves.empty();
        }
        if (!lhs.leaves.empty() && !rhs.leaves.empty()) {
            const int lhs_degree = graph.degrees[static_cast<size_t>(lhs_first)];
            const int rhs_degree = graph.degrees[static_cast<size_t>(rhs_first)];
            if (lhs_degree != rhs_degree) {
                return lhs_degree < rhs_degree;
            }
        }
        return lhs_first < rhs_first;
    });

    std::vector<int> order;
    order.reserve(static_cast<size_t>(graph.vertex_count));
    for (const RootLeaves& item : root_leaves) {
        const std::vector<int> ordered = order_subtree_vertices(graph, item.leaves, component_of_vertex);
        order.insert(order.end(), ordered.begin(), ordered.end());
    }
    return order;
}

void build_row_and_col_permutations(const CsrMatrix& matrix, ReorderPlan* plan) {
    plan->row_permutation.clear();
    plan->col_permutation.clear();
    plan->row_permutation.reserve(static_cast<size_t>(matrix.rows));
    plan->col_permutation.reserve(static_cast<size_t>(matrix.cols));
    plan->row_old_to_new.assign(static_cast<size_t>(matrix.rows), -1);
    plan->col_old_to_new.assign(static_cast<size_t>(matrix.cols), -1);

    for (int old_vertex : plan->vertex_order) {
        if (old_vertex < matrix.rows) {
            const int new_row = static_cast<int>(plan->row_permutation.size());
            plan->row_permutation.push_back(old_vertex);
            plan->row_old_to_new[static_cast<size_t>(old_vertex)] = new_row;
        }
        if (old_vertex < matrix.cols) {
            const int new_col = static_cast<int>(plan->col_permutation.size());
            plan->col_permutation.push_back(old_vertex);
            plan->col_old_to_new[static_cast<size_t>(old_vertex)] = new_col;
        }
    }
}

CsrMatrix apply_reorder_impl(const CsrMatrix& matrix, const ReorderPlan& plan) {
    if (static_cast<int>(plan.row_permutation.size()) != matrix.rows) {
        throw std::runtime_error("Row permutation size does not match matrix rows.");
    }
    if (static_cast<int>(plan.col_permutation.size()) != matrix.cols) {
        throw std::runtime_error("Column permutation size does not match matrix cols.");
    }

    CsrMatrix reordered;
    reordered.rows = matrix.rows;
    reordered.cols = matrix.cols;
    reordered.row_offsets.assign(static_cast<size_t>(matrix.rows) + 1, 0);
    reordered.col_indices.reserve(matrix.col_indices.size());
    reordered.values.reserve(matrix.values.size());

    for (int new_row = 0; new_row < matrix.rows; ++new_row) {
        const int old_row = plan.row_permutation[static_cast<size_t>(new_row)];
        const int begin = matrix.row_offsets[static_cast<size_t>(old_row)];
        const int end = matrix.row_offsets[static_cast<size_t>(old_row) + 1];

        std::vector<std::pair<int, float>> remapped_entries;
        remapped_entries.reserve(static_cast<size_t>(end - begin));
        for (int idx = begin; idx < end; ++idx) {
            const int old_col = matrix.col_indices[static_cast<size_t>(idx)];
            const int new_col = plan.col_old_to_new[static_cast<size_t>(old_col)];
            remapped_entries.emplace_back(new_col, matrix.values[static_cast<size_t>(idx)]);
        }

        std::sort(remapped_entries.begin(), remapped_entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        reordered.row_offsets[static_cast<size_t>(new_row) + 1] =
            reordered.row_offsets[static_cast<size_t>(new_row)] + static_cast<int>(remapped_entries.size());
        for (const auto& entry : remapped_entries) {
            reordered.col_indices.push_back(entry.first);
            reordered.values.push_back(entry.second);
        }
    }

    return reordered;
}

double modularity_gain_score(int shared_edges, int degree_v, int community_degree_sum, int edge_count) {
    if (shared_edges <= 0 || edge_count <= 0) {
        return -1.0;
    }
    const double two_m = 2.0 * static_cast<double>(edge_count);
    const double actual = static_cast<double>(shared_edges) / two_m;
    const double expected = (static_cast<double>(degree_v) * static_cast<double>(community_degree_sum)) / (two_m * two_m);
    return actual - expected;
}

}  // namespace

ReorderPlan build_affinity_reorder(const CsrMatrix& matrix, int tile_cols) {
    if (tile_cols <= 0) {
        throw std::runtime_error("tile_cols must be positive.");
    }

    ReorderPlan plan;
    const std::vector<RowFeature> original_features = build_feature_sequence(matrix, tile_cols);
    plan.original_metrics = compute_metrics_from_features(original_features, tile_cols);

    const GraphData graph = build_graph_from_matrix(matrix);
    std::vector<TreeNode> tree_nodes;
    DsuState dsu = make_dsu(graph, &tree_nodes);
    const std::vector<int> degree_order = build_degree_order(graph);

    std::vector<int> touched_roots;
    touched_roots.reserve(32);
    for (int vertex : degree_order) {
        const int vertex_root = dsu_find(&dsu, vertex);
        touched_roots.clear();
        std::vector<int> shared_edges;
        std::vector<int> candidate_roots;
        std::vector<int> seen_root_index(static_cast<size_t>(graph.vertex_count), -1);

        for (int neighbor : graph.neighbors[static_cast<size_t>(vertex)]) {
            const int neighbor_root = dsu_find(&dsu, neighbor);
            if (neighbor_root == vertex_root) {
                continue;
            }
            int index = seen_root_index[static_cast<size_t>(neighbor_root)];
            if (index < 0) {
                index = static_cast<int>(candidate_roots.size());
                seen_root_index[static_cast<size_t>(neighbor_root)] = index;
                candidate_roots.push_back(neighbor_root);
                shared_edges.push_back(0);
                touched_roots.push_back(neighbor_root);
            }
            shared_edges[static_cast<size_t>(index)] += 1;
        }

        double best_gain = 0.0;
        int best_root = -1;
        for (size_t idx = 0; idx < candidate_roots.size(); ++idx) {
            const int candidate_root = candidate_roots[idx];
            const double gain = modularity_gain_score(shared_edges[idx],
                                                      graph.degrees[static_cast<size_t>(vertex)],
                                                      dsu.community_degree_sum[static_cast<size_t>(candidate_root)],
                                                      graph.edge_count);
            if (gain > best_gain ||
                (std::fabs(gain - best_gain) < 1.0e-12 &&
                 dsu.community_degree_sum[static_cast<size_t>(candidate_root)] <
                     dsu.community_degree_sum[static_cast<size_t>(best_root >= 0 ? best_root : candidate_root)]) ||
                (std::fabs(gain - best_gain) < 1.0e-12 &&
                 dsu.community_degree_sum[static_cast<size_t>(candidate_root)] ==
                     dsu.community_degree_sum[static_cast<size_t>(best_root >= 0 ? best_root : candidate_root)] &&
                 candidate_root < (best_root >= 0 ? best_root : candidate_root))) {
                best_gain = gain;
                best_root = candidate_root;
            }
        }

        for (int touched_root : touched_roots) {
            seen_root_index[static_cast<size_t>(touched_root)] = -1;
        }

        if (best_root >= 0 && best_gain > 0.0) {
            (void)dsu_union(&dsu, vertex_root, best_root, &tree_nodes);
        }
    }

    plan.vertex_order = build_vertex_order(graph, tree_nodes, &dsu);
    build_row_and_col_permutations(matrix, &plan);

    const CsrMatrix reordered_matrix = apply_reorder_impl(matrix, plan);
    const std::vector<RowFeature> reordered_features = build_feature_sequence(reordered_matrix, tile_cols);
    plan.reordered_metrics = compute_metrics_from_features(reordered_features, tile_cols);
    return plan;
}

CsrMatrix apply_reorder(const CsrMatrix& matrix, const ReorderPlan& plan) {
    return apply_reorder_impl(matrix, plan);
}

DenseMatrix apply_rhs_reorder(const DenseMatrix& rhs, const ReorderPlan& plan) {
    if (static_cast<int>(plan.col_permutation.size()) != rhs.rows) {
        throw std::runtime_error("Column permutation size does not match rhs rows.");
    }

    DenseMatrix reordered;
    reordered.rows = rhs.rows;
    reordered.cols = rhs.cols;
    reordered.values.assign(rhs.values.size(), 0.0f);

    for (int new_row = 0; new_row < rhs.rows; ++new_row) {
        const int old_row = plan.col_permutation[static_cast<size_t>(new_row)];
        for (int col = 0; col < rhs.cols; ++col) {
            reordered(new_row, col) = rhs(old_row, col);
        }
    }
    return reordered;
}

DenseMatrix restore_output_row_order(const DenseMatrix& reordered_output, const ReorderPlan& plan) {
    if (static_cast<int>(plan.row_permutation.size()) != reordered_output.rows) {
        throw std::runtime_error("Row permutation size does not match output rows.");
    }

    DenseMatrix restored;
    restored.rows = reordered_output.rows;
    restored.cols = reordered_output.cols;
    restored.values.assign(reordered_output.values.size(), 0.0f);

    for (int new_row = 0; new_row < reordered_output.rows; ++new_row) {
        const int old_row = plan.row_permutation[static_cast<size_t>(new_row)];
        for (int col = 0; col < reordered_output.cols; ++col) {
            restored(old_row, col) = reordered_output(new_row, col);
        }
    }
    return restored;
}

ReorderDecision evaluate_reorder_decision(const ReorderPlan& plan,
                                          const BittcfFormat& original_format,
                                          const BittcfFormat& reordered_format) {
    ReorderDecision decision;

    auto add_vote = [&](bool positive, double weight, const char* good_reason, const char* bad_reason) {
        decision.score += positive ? weight : -weight;
        if (decision.reason.empty()) {
            decision.reason = positive ? good_reason : bad_reason;
        }
    };

    const bool block_improved =
        reordered_format.tc_block_count > 0 &&
        reordered_format.tc_block_count <= static_cast<int>(0.98 * static_cast<double>(original_format.tc_block_count));
    add_vote(block_improved, 3.0, "BitTCF block count decreases.", "BitTCF block count does not decrease enough.");

    const bool occupancy_improved =
        reordered_format.avg_block_occupancy >= original_format.avg_block_occupancy * 1.02;
    add_vote(occupancy_improved, 2.5, "BitTCF block occupancy improves.", "BitTCF block occupancy does not improve enough.");

    const bool compression_improved =
        reordered_format.compression_ratio_vs_csr >= original_format.compression_ratio_vs_csr * 1.02;
    add_vote(compression_improved, 2.0, "BitTCF compression improves.", "BitTCF compression does not improve enough.");

    const bool locality_improved =
        plan.reordered_metrics.adjacent_tile_jaccard >= plan.original_metrics.adjacent_tile_jaccard * 1.02;
    add_vote(locality_improved, 1.0, "Adjacent tile similarity improves.", "Adjacent tile similarity does not improve enough.");

    const bool tiles_reduce =
        plan.reordered_metrics.avg_unique_col_tiles <= plan.original_metrics.avg_unique_col_tiles * 0.99;
    add_vote(tiles_reduce, 1.0, "Average unique tile count decreases.", "Average unique tile count does not decrease enough.");

    const bool row_tile_occupancy_improves =
        plan.reordered_metrics.avg_tile_occupancy >= plan.original_metrics.avg_tile_occupancy * 1.01;
    add_vote(row_tile_occupancy_improves, 1.0, "Row-level tile occupancy improves.", "Row-level tile occupancy does not improve enough.");

    decision.apply_reorder = block_improved && occupancy_improved && compression_improved && decision.score >= 4.0;
    if (decision.apply_reorder) {
        decision.reason = "Reorder enabled: BitTCF and locality metrics improve together.";
    }
    return decision;
}

}  // namespace acc_spmm

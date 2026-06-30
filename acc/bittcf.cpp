#include "acc/bittcf.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace acc_spmm {
namespace {

struct BlockBuilder {
    explicit BlockBuilder(int words)
        : bit_words(static_cast<size_t>(words), 0ULL) {}

    std::vector<unsigned long long> bit_words;
    std::vector<int> columns;
    int nnz = 0;

    void add(int local_row, int local_col, int global_col, int tile_cols) {
        const int linear_index = local_row * tile_cols + local_col;
        const int word_index = linear_index / 64;
        const int bit_index = linear_index % 64;
        bit_words[static_cast<size_t>(word_index)] |= (1ULL << bit_index);
        columns.push_back(global_col);
        nnz += 1;
    }
};

double csr_storage_bytes(const CsrMatrix& matrix) {
    const double row_offsets_bytes = static_cast<double>(matrix.row_offsets.size()) * sizeof(int);
    const double col_indices_bytes = static_cast<double>(matrix.col_indices.size()) * sizeof(int);
    const double values_bytes = static_cast<double>(matrix.values.size()) * sizeof(float);
    return row_offsets_bytes + col_indices_bytes + values_bytes;
}

}  // namespace

BittcfFormat build_bittcf(const CsrMatrix& matrix, int tile_rows, int tile_cols) {
    if (tile_rows <= 0 || tile_cols <= 0) {
        throw std::runtime_error("BitTCF tile sizes must be positive.");
    }

    BittcfFormat format;
    format.tile_rows = tile_rows;
    format.tile_cols = tile_cols;
    format.nnz = matrix.nnz();
    format.words_per_block = (tile_rows * tile_cols + 63) / 64;
    format.row_window_count = (matrix.rows + tile_rows - 1) / tile_rows;
    format.row_window_offsets.reserve(static_cast<size_t>(format.row_window_count) + 1);
    format.row_window_offsets.push_back(0);

    double occupancy_sum = 0.0;
    for (int window = 0; window < format.row_window_count; ++window) {
        const int row_begin = window * tile_rows;
        const int row_end = std::min(matrix.rows, row_begin + tile_rows);
        std::map<int, BlockBuilder> blocks;

        for (int row = row_begin; row < row_end; ++row) {
            const int local_row = row - row_begin;
            const int begin = matrix.row_offsets[static_cast<size_t>(row)];
            const int end = matrix.row_offsets[static_cast<size_t>(row) + 1];
            for (int idx = begin; idx < end; ++idx) {
                const int col = matrix.col_indices[static_cast<size_t>(idx)];
                const int tile_col = col / tile_cols;
                auto block_it = blocks.try_emplace(tile_col, format.words_per_block).first;
                block_it->second.add(local_row, col % tile_cols, col, tile_cols);
            }
        }

        for (auto& entry : blocks) {
            format.tc_block_offsets.push_back(entry.first);
            format.original_columns.insert(format.original_columns.end(), entry.second.columns.begin(), entry.second.columns.end());
            format.local_bitmasks.insert(format.local_bitmasks.end(),
                                         entry.second.bit_words.begin(),
                                         entry.second.bit_words.end());
            occupancy_sum += static_cast<double>(entry.second.nnz) /
                             static_cast<double>(tile_rows * tile_cols);
            format.tc_block_count += 1;
        }

        format.row_window_offsets.push_back(format.tc_block_count);
    }

    if (format.tc_block_count > 0) {
        format.avg_block_occupancy = occupancy_sum / static_cast<double>(format.tc_block_count);
    }

    const double bittcf_bytes = static_cast<double>(format.row_window_offsets.size()) * sizeof(int) +
                                static_cast<double>(format.tc_block_offsets.size()) * sizeof(int) +
                                static_cast<double>(format.original_columns.size()) * sizeof(int) +
                                static_cast<double>(format.local_bitmasks.size()) * sizeof(unsigned long long);
    const double csr_bytes = csr_storage_bytes(matrix);
    if (bittcf_bytes > 0.0) {
        format.compression_ratio_vs_csr = csr_bytes / bittcf_bytes;
    }

    return format;
}

}  // namespace acc_spmm

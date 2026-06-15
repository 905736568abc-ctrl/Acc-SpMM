#pragma once

#include <string>
#include <vector>

namespace acc_spmm {

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;

    [[nodiscard]] int nnz() const {
        return static_cast<int>(values.size());
    }

    [[nodiscard]] bool empty() const {
        return rows == 0 || cols == 0 || values.empty();
    }
};

struct DenseMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> values;

    [[nodiscard]] size_t size() const {
        return values.size();
    }

    float& operator()(int row, int col) {
        return values[static_cast<size_t>(row) * cols + col];
    }

    float operator()(int row, int col) const {
        return values[static_cast<size_t>(row) * cols + col];
    }
};

struct BenchmarkConfig {
    std::string matrix_path;
    int dense_cols = 128;
    int warmup = 5;
    int repeat = 20;
    bool check = true;
};

struct KernelResult {
    float average_ms = 0.0f;
    double gflops = 0.0;
    DenseMatrix output;
};

struct ReorderPlan {
    std::vector<int> permutation;
};

struct BittcfFormat {
    int tile_rows = 16;
    int tile_cols = 16;
    int nnz = 0;
    std::vector<int> row_window_offsets;
    std::vector<int> tc_block_offsets;
    std::vector<int> original_columns;
    std::vector<unsigned long long> local_bitmasks;
};

struct PipelineConfig {
    int stages = 2;
    int block_threads = 128;
    bool use_cp_async = true;
};

inline BittcfFormat build_bittcf(const CsrMatrix& matrix, int tile_rows = 16, int tile_cols = 16) {
    BittcfFormat format;
    format.tile_rows = tile_rows;
    format.tile_cols = tile_cols;
    format.nnz = matrix.nnz();
    format.row_window_offsets.push_back(0);
    format.tc_block_offsets.push_back(0);
    return format;
}

}  // namespace acc_spmm

#include "benchmark/matrix_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace acc_spmm {
namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

}  // namespace

CsrMatrix load_matrix_market(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open matrix file: " + path);
    }

    std::string header;
    if (!std::getline(input, header)) {
        throw std::runtime_error("Failed to read Matrix Market header from: " + path);
    }

    const std::string lowered = to_lower(header);
    const bool symmetric = lowered.find("symmetric") != std::string::npos;
    const bool pattern = lowered.find("pattern") != std::string::npos;

    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }
    if (line.empty()) {
        throw std::runtime_error("Matrix Market size line is missing: " + path);
    }

    std::istringstream size_stream(line);
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    size_stream >> rows >> cols >> nnz;
    if (rows <= 0 || cols <= 0 || nnz < 0) {
        throw std::runtime_error("Invalid Matrix Market dimensions in: " + path);
    }

    std::vector<std::tuple<int, int, float>> entries;
    entries.reserve(symmetric ? static_cast<size_t>(nnz) * 2 : static_cast<size_t>(nnz));

    int row = 0;
    int col = 0;
    float value = 1.0f;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        std::istringstream entry_stream(line);
        if (pattern) {
            entry_stream >> row >> col;
            value = 1.0f;
        } else {
            entry_stream >> row >> col >> value;
        }
        if (!entry_stream) {
            throw std::runtime_error("Invalid Matrix Market entry in: " + path);
        }
        --row;
        --col;
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::runtime_error("Matrix Market entry index out of range in: " + path);
        }
        entries.emplace_back(row, col, value);
        if (symmetric && row != col) {
            entries.emplace_back(col, row, value);
        }
    }

    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        if (std::get<0>(lhs) != std::get<0>(rhs)) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        }
        return std::get<1>(lhs) < std::get<1>(rhs);
    });

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_offsets.assign(static_cast<size_t>(rows) + 1, 0);
    matrix.col_indices.reserve(entries.size());
    matrix.values.reserve(entries.size());

    for (const auto& entry : entries) {
        ++matrix.row_offsets[static_cast<size_t>(std::get<0>(entry)) + 1];
        matrix.col_indices.push_back(std::get<1>(entry));
        matrix.values.push_back(std::get<2>(entry));
    }

    for (int i = 0; i < rows; ++i) {
        matrix.row_offsets[static_cast<size_t>(i) + 1] += matrix.row_offsets[static_cast<size_t>(i)];
    }

    return matrix;
}

DenseMatrix make_dense_matrix(int rows, int cols, float seed) {
    DenseMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.values.resize(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            matrix(r, c) = seed * static_cast<float>(((r * 131 + c * 17) % 97) + 1);
        }
    }
    return matrix;
}

DenseMatrix compute_reference_spmm(const CsrMatrix& matrix, const DenseMatrix& rhs) {
    if (rhs.rows != matrix.cols) {
        throw std::runtime_error("Reference SpMM dimension mismatch.");
    }

    DenseMatrix out;
    out.rows = matrix.rows;
    out.cols = rhs.cols;
    out.values.assign(static_cast<size_t>(out.rows) * out.cols, 0.0f);

    for (int row = 0; row < matrix.rows; ++row) {
        const int begin = matrix.row_offsets[static_cast<size_t>(row)];
        const int end = matrix.row_offsets[static_cast<size_t>(row) + 1];
        for (int idx = begin; idx < end; ++idx) {
            const int col_idx = matrix.col_indices[static_cast<size_t>(idx)];
            const float value_a = matrix.values[static_cast<size_t>(idx)];
            for (int n = 0; n < rhs.cols; ++n) {
                out(row, n) += value_a * rhs(col_idx, n);
            }
        }
    }

    return out;
}

float max_abs_diff(const DenseMatrix& lhs, const DenseMatrix& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols || lhs.values.size() != rhs.values.size()) {
        throw std::runtime_error("Dense matrix diff dimension mismatch.");
    }

    float diff = 0.0f;
    for (size_t i = 0; i < lhs.values.size(); ++i) {
        diff = std::max(diff, std::fabs(lhs.values[i] - rhs.values[i]));
    }
    return diff;
}

double matrix_density(const CsrMatrix& matrix) {
    if (matrix.rows <= 0 || matrix.cols <= 0) {
        return 0.0;
    }
    const double total = static_cast<double>(matrix.rows) * static_cast<double>(matrix.cols);
    return static_cast<double>(matrix.nnz()) / total;
}

double spmm_flop_count(const CsrMatrix& matrix, int dense_cols) {
    return 2.0 * static_cast<double>(matrix.nnz()) * static_cast<double>(dense_cols);
}

}  // namespace acc_spmm

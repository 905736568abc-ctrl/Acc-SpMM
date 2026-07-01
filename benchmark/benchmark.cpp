#include "acc/reorder.hpp"
#include "baseline/cusparse_spmm.hpp"
#include "benchmark/matrix_loader.hpp"
#include "dtc/dtc_spmm.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace acc_spmm {
namespace {

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for argument: ") + name);
            }
            return argv[++i];
        };

        if (arg == "--matrix") {
            config.matrix_path = require_value("--matrix");
        } else if (arg == "--n") {
            config.dense_cols = std::stoi(require_value("--n"));
        } else if (arg == "--warmup") {
            config.warmup = std::stoi(require_value("--warmup"));
        } else if (arg == "--repeat") {
            config.repeat = std::stoi(require_value("--repeat"));
        } else if (arg == "--check") {
            config.check = std::stoi(require_value("--check")) != 0;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.matrix_path.empty()) {
        throw std::runtime_error(
            "Usage: acc_spmm_bench --matrix path/to/matrix.mtx [--n 128] [--warmup 5] [--repeat 20] [--check 1]");
    }
    if (config.dense_cols <= 0 || config.warmup < 0 || config.repeat <= 0) {
        throw std::runtime_error("Invalid numeric arguments.");
    }
    return config;
}

}  // namespace
}  // namespace acc_spmm

int main(int argc, char** argv) {
    try {
        const acc_spmm::BenchmarkConfig config = acc_spmm::parse_args(argc, argv);
        const acc_spmm::CsrMatrix matrix = acc_spmm::load_matrix_market(config.matrix_path);
        const double density = acc_spmm::matrix_density(matrix);
        const double flop_count = acc_spmm::spmm_flop_count(matrix, config.dense_cols);
        const acc_spmm::DenseMatrix rhs = acc_spmm::make_dense_matrix(matrix.cols, config.dense_cols);
        const acc_spmm::ReorderPlan reorder = acc_spmm::build_affinity_reorder(matrix);
        const acc_spmm::CsrMatrix reordered_matrix = acc_spmm::apply_reorder(matrix, reorder);
        const acc_spmm::BittcfFormat bittcf_original = acc_spmm::build_bittcf(matrix);
        const acc_spmm::BittcfFormat bittcf_reordered = acc_spmm::build_bittcf(reordered_matrix);

        std::cout << "[Acc-SpMM-Reproduce]\n";
        std::cout << "matrix=" << config.matrix_path << "\n";
        std::cout << "rows=" << matrix.rows << ", cols=" << matrix.cols << ", nnz=" << matrix.nnz() << "\n";
        std::cout << "dense_cols=" << config.dense_cols << ", warmup=" << config.warmup << ", repeat=" << config.repeat
                  << "\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "density=" << density << "\n";
        std::cout << "spmm_flop_count=" << flop_count << "\n";
        std::cout << "reorder_plan_size=" << reorder.row_permutation.size() << "\n";
        std::cout << "reorder_before adjacent_tile_jaccard=" << reorder.original_metrics.adjacent_tile_jaccard
                  << " adjacent_first_tile_distance=" << reorder.original_metrics.adjacent_first_tile_distance
                  << " avg_unique_col_tiles=" << reorder.original_metrics.avg_unique_col_tiles
                  << " avg_tile_occupancy=" << reorder.original_metrics.avg_tile_occupancy << "\n";
        std::cout << "reorder_after adjacent_tile_jaccard=" << reorder.reordered_metrics.adjacent_tile_jaccard
                  << " adjacent_first_tile_distance=" << reorder.reordered_metrics.adjacent_first_tile_distance
                  << " avg_unique_col_tiles=" << reorder.reordered_metrics.avg_unique_col_tiles
                  << " avg_tile_occupancy=" << reorder.reordered_metrics.avg_tile_occupancy << "\n";
        std::cout << "bittcf_original row_windows=" << bittcf_original.row_window_count
                  << " tc_blocks=" << bittcf_original.tc_block_count
                  << " words_per_block=" << bittcf_original.words_per_block
                  << " avg_block_occupancy=" << bittcf_original.avg_block_occupancy
                  << " compression_ratio_vs_csr=" << bittcf_original.compression_ratio_vs_csr << "\n";
        std::cout << "bittcf_reordered row_windows=" << bittcf_reordered.row_window_count
                  << " tc_blocks=" << bittcf_reordered.tc_block_count
                  << " words_per_block=" << bittcf_reordered.words_per_block
                  << " avg_block_occupancy=" << bittcf_reordered.avg_block_occupancy
                  << " compression_ratio_vs_csr=" << bittcf_reordered.compression_ratio_vs_csr << "\n";

        acc_spmm::DenseMatrix reference;
        float diff = -1.0f;
        float reference_max_abs = -1.0f;
        float relative_diff = -1.0f;
        if (config.check) {
            std::cout << "computing_reference=1\n";
            reference = acc_spmm::compute_reference_spmm(matrix, rhs);
            reference_max_abs = acc_spmm::max_abs_value(reference);
        }

        const acc_spmm::KernelResult cusparse = acc_spmm::run_cusparse_spmm(matrix, rhs, config.warmup, config.repeat);
        std::cout << "kernel=cusparse average_ms=" << cusparse.average_ms << " gflops=" << cusparse.gflops << "\n";

        if (config.check) {
            diff = acc_spmm::max_abs_diff(reference, cusparse.output);
            relative_diff = acc_spmm::max_relative_diff(reference, cusparse.output);
            std::cout << "kernel=cusparse reference_max_abs=" << reference_max_abs << "\n";
            std::cout << "kernel=cusparse max_abs_diff=" << diff << "\n";
            std::cout << "kernel=cusparse max_relative_diff=" << relative_diff << "\n";
        }

        std::cout << "csv,matrix=" << config.matrix_path << ",rows=" << matrix.rows << ",cols=" << matrix.cols
                  << ",nnz=" << matrix.nnz() << ",density=" << density << ",n=" << config.dense_cols
                  << ",warmup=" << config.warmup << ",repeat=" << config.repeat << ",kernel=cusparse"
                  << ",reorder_before_jaccard=" << reorder.original_metrics.adjacent_tile_jaccard
                  << ",reorder_after_jaccard=" << reorder.reordered_metrics.adjacent_tile_jaccard
                  << ",reorder_before_tile_distance=" << reorder.original_metrics.adjacent_first_tile_distance
                  << ",reorder_after_tile_distance=" << reorder.reordered_metrics.adjacent_first_tile_distance
                  << ",bittcf_original_blocks=" << bittcf_original.tc_block_count
                  << ",bittcf_reordered_blocks=" << bittcf_reordered.tc_block_count
                  << ",bittcf_original_occupancy=" << bittcf_original.avg_block_occupancy
                  << ",bittcf_reordered_occupancy=" << bittcf_reordered.avg_block_occupancy
                  << ",bittcf_original_compression=" << bittcf_original.compression_ratio_vs_csr
                  << ",bittcf_reordered_compression=" << bittcf_reordered.compression_ratio_vs_csr
                  << ",average_ms=" << cusparse.average_ms << ",gflops=" << cusparse.gflops
                  << ",reference_max_abs=" << reference_max_abs << ",max_abs_diff=" << diff
                  << ",max_relative_diff=" << relative_diff << "\n";

        try {
            (void)acc_spmm::run_dtc_placeholder(matrix, rhs, config.warmup, config.repeat);
        } catch (const std::exception& ex) {
            std::cout << "kernel=dtc status=skipped reason=\"" << ex.what() << "\"\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}

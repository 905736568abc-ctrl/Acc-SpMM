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
        const acc_spmm::DenseMatrix rhs = acc_spmm::make_dense_matrix(matrix.cols, config.dense_cols);
        const acc_spmm::ReorderPlan reorder = acc_spmm::build_affinity_reorder(matrix);
        const double density = acc_spmm::matrix_density(matrix);
        const double flop_count = acc_spmm::spmm_flop_count(matrix, config.dense_cols);

        std::cout << "[Acc-SpMM-Reproduce]\n";
        std::cout << "matrix=" << config.matrix_path << "\n";
        std::cout << "rows=" << matrix.rows << ", cols=" << matrix.cols << ", nnz=" << matrix.nnz() << "\n";
        std::cout << "dense_cols=" << config.dense_cols << ", warmup=" << config.warmup << ", repeat=" << config.repeat
                  << "\n";
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "density=" << density << "\n";
        std::cout << "spmm_flop_count=" << flop_count << "\n";
        std::cout << "reorder_plan_size=" << reorder.permutation.size() << "\n";

        acc_spmm::DenseMatrix reference;
        float diff = -1.0f;
        if (config.check) {
            std::cout << "computing_reference=1\n";
            reference = acc_spmm::compute_reference_spmm(matrix, rhs);
        }

        const acc_spmm::KernelResult cusparse = acc_spmm::run_cusparse_spmm(matrix, rhs, config.warmup, config.repeat);
        std::cout << "kernel=cusparse average_ms=" << cusparse.average_ms << " gflops=" << cusparse.gflops << "\n";

        if (config.check) {
            diff = acc_spmm::max_abs_diff(reference, cusparse.output);
            std::cout << "kernel=cusparse max_abs_diff=" << diff << "\n";
        }

        std::cout << "csv,matrix=" << config.matrix_path << ",rows=" << matrix.rows << ",cols=" << matrix.cols
                  << ",nnz=" << matrix.nnz() << ",density=" << density << ",n=" << config.dense_cols
                  << ",warmup=" << config.warmup << ",repeat=" << config.repeat << ",kernel=cusparse"
                  << ",average_ms=" << cusparse.average_ms << ",gflops=" << cusparse.gflops
                  << ",max_abs_diff=" << diff << "\n";

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

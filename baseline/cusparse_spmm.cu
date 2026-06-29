#include "baseline/cusparse_spmm.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace acc_spmm {
namespace {

void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

void check_cusparse(cusparseStatus_t status, const char* message) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(message) + ": cuSPARSE failure");
    }
}

}  // namespace

KernelResult run_cusparse_spmm(const CsrMatrix& matrix, const DenseMatrix& rhs, int warmup, int repeat) {
    if (matrix.empty()) {
        throw std::runtime_error("Input CSR matrix is empty.");
    }
    if (rhs.rows != matrix.cols) {
        throw std::runtime_error("Dense rhs row count must match sparse matrix column count.");
    }
    if (warmup < 0 || repeat <= 0) {
        throw std::runtime_error("Warmup must be non-negative and repeat must be positive.");
    }

    DenseMatrix out;
    out.rows = matrix.rows;
    out.cols = rhs.cols;
    out.values.assign(static_cast<size_t>(out.rows) * out.cols, 0.0f);

    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    float* d_values = nullptr;
    float* d_rhs = nullptr;
    float* d_out = nullptr;
    void* d_buffer = nullptr;
    size_t buffer_size = 0;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseDnMatDescr_t mat_b = nullptr;
    cusparseDnMatDescr_t mat_c = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_row_offsets), matrix.row_offsets.size() * sizeof(int)), "cudaMalloc row_offsets");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_col_indices), matrix.col_indices.size() * sizeof(int)), "cudaMalloc col_indices");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_values), matrix.values.size() * sizeof(float)), "cudaMalloc values");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_rhs), rhs.values.size() * sizeof(float)), "cudaMalloc rhs");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out), out.values.size() * sizeof(float)), "cudaMalloc out");

    check_cuda(cudaMemcpy(d_row_offsets, matrix.row_offsets.data(), matrix.row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "Memcpy row_offsets");
    check_cuda(cudaMemcpy(d_col_indices, matrix.col_indices.data(), matrix.col_indices.size() * sizeof(int), cudaMemcpyHostToDevice), "Memcpy col_indices");
    check_cuda(cudaMemcpy(d_values, matrix.values.data(), matrix.values.size() * sizeof(float), cudaMemcpyHostToDevice), "Memcpy values");
    check_cuda(cudaMemcpy(d_rhs, rhs.values.data(), rhs.values.size() * sizeof(float), cudaMemcpyHostToDevice), "Memcpy rhs");
    check_cuda(cudaMemset(d_out, 0, out.values.size() * sizeof(float)), "Memset out");

    check_cusparse(cusparseCreate(&handle), "cusparseCreate");
    check_cusparse(
        cusparseCreateCsr(&mat_a,
                          matrix.rows,
                          matrix.cols,
                          matrix.nnz(),
                          d_row_offsets,
                          d_col_indices,
                          d_values,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_32F),
        "cusparseCreateCsr");
    check_cusparse(
        cusparseCreateDnMat(&mat_b, rhs.rows, rhs.cols, rhs.cols, d_rhs, CUDA_R_32F, CUSPARSE_ORDER_ROW),
        "cusparseCreateDnMat B");
    check_cusparse(
        cusparseCreateDnMat(&mat_c, out.rows, out.cols, out.cols, d_out, CUDA_R_32F, CUSPARSE_ORDER_ROW),
        "cusparseCreateDnMat C");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cusparse(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           mat_a,
                                           mat_b,
                                           &beta,
                                           mat_c,
                                           CUDA_R_32F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           &buffer_size),
                   "cusparseSpMM_bufferSize");
    check_cuda(cudaMalloc(&d_buffer, buffer_size), "cudaMalloc buffer");

    auto launch_once = [&]() {
        check_cusparse(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    mat_a,
                                    mat_b,
                                    &beta,
                                    mat_c,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    d_buffer),
                       "cusparseSpMM");
    };

    for (int i = 0; i < warmup; ++i) {
        check_cuda(cudaMemset(d_out, 0, out.values.size() * sizeof(float)), "Memset out warmup");
        launch_once();
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < repeat; ++i) {
        check_cuda(cudaMemset(d_out, 0, out.values.size() * sizeof(float)), "Memset out repeat");
        launch_once();
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
    check_cuda(cudaMemcpy(out.values.data(), d_out, out.values.size() * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy out");

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check_cusparse(cusparseDestroyDnMat(mat_b), "cusparseDestroyDnMat B");
    check_cusparse(cusparseDestroyDnMat(mat_c), "cusparseDestroyDnMat C");
    check_cusparse(cusparseDestroySpMat(mat_a), "cusparseDestroySpMat");
    check_cusparse(cusparseDestroy(handle), "cusparseDestroy");
    check_cuda(cudaFree(d_buffer), "cudaFree buffer");
    check_cuda(cudaFree(d_out), "cudaFree out");
    check_cuda(cudaFree(d_rhs), "cudaFree rhs");
    check_cuda(cudaFree(d_values), "cudaFree values");
    check_cuda(cudaFree(d_col_indices), "cudaFree col_indices");
    check_cuda(cudaFree(d_row_offsets), "cudaFree row_offsets");

    KernelResult result;
    result.average_ms = total_ms / static_cast<float>(repeat);
    result.gflops =
        (2.0 * static_cast<double>(matrix.nnz()) * static_cast<double>(rhs.cols)) /
        (static_cast<double>(result.average_ms) * 1.0e6);
    result.output = std::move(out);
    return result;
}

}  // namespace acc_spmm

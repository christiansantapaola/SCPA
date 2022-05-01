extern "C" {
#include "CSRMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}
#include "cudaUtils.cuh"
#include <cuda.h>
#include "SpMVKernel.cuh"

#define MAX_X_SIZE 65536 / sizeof(float)

/*
 * Calcolo Performance:
 * Accessi alla memoria globale:
 * int row = blockIdx.x * blockDim.x + threadIdx.x; + 1
 * int row_start = row_ptr[row];                    + 1
 * int row_end = row_ptr[row + 1];                  + 1
 * data[elem]                                       + 1
 * x[col_index[elem]]                               + 2
 * y[row]                                           + 1
 * Totale                                           + 7
 * Numero Operazioni Float:
 * dot += data[elem] * x[col_index[elem]];          + 2
 * y[row] += dot;                                   + 1
 * Totale                                           + 3
 * Ratio float/access = 7/3
 */

__global__ void
SpMV_CSR_kernel(u_int64_t num_rows, const float *data, const u_int64_t *col_index, const u_int64_t *row_ptr, const float *x, float *y) {
    u_int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        u_int64_t row_start = row_ptr[row];
        u_int64_t row_end = row_ptr[row + 1];
        for (u_int64_t elem = row_start; elem < row_end; elem++) {
            dot += data[elem] * x[col_index[elem]];
        }
        y[row] += dot;
    }
}

extern "C"
int CSRMatrix_SpMV_CUDA(const CSRMatrix *d_matrix, const Vector *d_x, Vector *d_y, SpMVResultCUDA *result) {
    cudaEvent_t start, stop;
    size_t memoryUsed;
    if (!d_matrix || !d_x || !d_y) {
        return SPMV_FAIL;
    }
    if (d_x->size != d_matrix->col_size && d_y->size != d_matrix->row_size) {
        return SPMV_FAIL;
    }
    memoryUsed = (d_matrix->num_non_zero_elements + d_x->size + d_y->size) * sizeof(float) + sizeof(u_int64_t) * (d_matrix->row_size + 1 + d_matrix->num_non_zero_elements);
    int bestDev = CudaUtils_getBestDevice(memoryUsed);
    if (bestDev == -1) {
        return SPMV_FAIL;
    }
    CudaUtils_setDevice(bestDev);
    cudaDeviceProp prop;
    BlockGridInfo blockGridInfo;
    CudaUtils_getDeviceProp(bestDev, &prop);
    CudaUtils_getBestCudaParameters(d_matrix->row_size, &prop, &blockGridInfo);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    SpMV_CSR_kernel<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(d_matrix->row_size,
                                                                         d_matrix->data,
                                                                         d_matrix->col_index,
                                                                         d_matrix->row_pointer,
                                                                         d_x->data,
                                                                         d_y->data);
    cudaEventRecord(stop);

    if (result) {
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result->GPUKernelExecutionTime, start, stop);
    }
    return SPMV_SUCCESS;
}
#include <cuda.h>

#include "cudaUtils.cuh"
extern "C" {
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}
#include "SpMVKernel.cuh"

__global__ void SpMV_ELL_kernel(u_int64_t num_rows, const float *data, const u_int64_t *col_index, u_int64_t num_elem, const float *x, float *y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        for (int i = 0; i < num_elem; i++) {
            int index = row + num_rows * i;
            dot += data[index] * x[col_index[index]];
        }
        y[row] += dot;
    }
}

int ELLMatrix_SpMV_CUDA(const ELLMatrix *matrix, const Vector *x, Vector *y, SpMVResultCUDA *result) {
    cudaEvent_t start, stop;
    size_t memoryUsed;
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        return SPMV_FAIL;
    }
    if (result) {
        memset(result, 0, sizeof(*result));
    }
    memoryUsed = (matrix->data_size + x->size + y->size) * sizeof(float) +   sizeof(u_int64_t) * (matrix->data_size);
    int bestDev = CudaUtils_getBestDevice(memoryUsed);
    if (bestDev == -1) {
        return SPMV_FAIL;
    }
    CudaUtils_setDevice(bestDev);
    cudaDeviceProp prop;
    BlockGridInfo blockGridInfo;
    CudaUtils_getDeviceProp(bestDev, &prop);
    CudaUtils_getBestCudaParameters(matrix->row_size, &prop, &blockGridInfo);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    SpMV_ELL_kernel<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(matrix->row_size, matrix->data, matrix->col_index, matrix->num_elem, x->data, y->data);
    cudaEventRecord(stop);
    if (result) {
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result->GPUKernelExecutionTime, start, stop);
    }
    return SPMV_SUCCESS;
}
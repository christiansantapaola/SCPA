//
// Created by 9669c on 11/03/2022.
//


extern "C" {
#include "CSRMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}
#include "cudaUtils.cuh"
#include <cuda.h>

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
void CSRMatrix_SpMV_GPU(const CSRMatrix *matrix, const Vector *x, Vector *y, SpMVResultCUDA *result) {
    float *d_matrix_data, *d_x, *d_y;
    u_int64_t *d_col_index, *d_row_ptr;
    cudaEvent_t start, stop, instart, instop, outstart, outstop;
    size_t memoryUsed;
    clock_t cstart = clock();
    if (!matrix || !x || !y) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (result) {
        memset(result, 0, sizeof(*result));
    }
    memoryUsed = (matrix->num_non_zero_elements + x->size + y->size) * sizeof(float) +   sizeof(u_int64_t) * (matrix->row_size + 1 + matrix->num_non_zero_elements);
    int bestDev = CudaUtils_getBestDevice();
    CudaUtils_setDevice(bestDev);
    cudaDeviceProp prop = {0};
    BlockGridInfo blockGridInfo;
    CudaUtils_getDeviceProp(bestDev, &prop);
    CudaUtils_getBestCudaParameters(matrix->row_size, &prop, &blockGridInfo);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&instart);
    cudaEventCreate(&instop);
    cudaEventCreate(&outstart);
    cudaEventCreate(&outstop);


    cudaEventRecord(instart);

    checkCudaErrors(cudaMalloc(&d_matrix_data, matrix->num_non_zero_elements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_col_index, matrix->num_non_zero_elements * sizeof(u_int64_t)));
    checkCudaErrors(cudaMalloc(&d_row_ptr, (matrix->row_size + 1) * sizeof(u_int64_t)));
    checkCudaErrors(cudaMalloc(&d_x, matrix->row_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y, matrix->row_size * sizeof(float)));

    checkCudaErrors(cudaMemcpyAsync(d_matrix_data, matrix->data,matrix->num_non_zero_elements * sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_col_index, matrix->col_index, matrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_row_ptr, matrix->row_pointer, (matrix->row_size + 1) * sizeof(u_int64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_x, x->data, matrix->row_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_y, y->data, matrix->row_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(instop);
    cudaEventRecord(start);
    SpMV_CSR_kernel<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(matrix->row_size,
                                d_matrix_data,
                                d_col_index,
                                d_row_ptr,
                                d_x,
                                d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop);

    cudaEventRecord(outstart);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaMemcpy(y->data, d_y, matrix->row_size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_matrix_data));
    checkCudaErrors(cudaFree(d_col_index));
    checkCudaErrors(cudaFree(d_row_ptr));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    cudaEventRecord(outstop);
    cudaEventSynchronize(stop);
    clock_t cend = clock();
    if (result) {
        result->success = 1;
        result->CPUFunctionExecutionTime = (((float)(cend - cstart)) / CLOCKS_PER_SEC) * 1000.0f;
        cudaEventElapsedTime(&result->GPUKernelExecutionTime, start, stop);
        cudaEventSynchronize(instop);
        cudaEventElapsedTime(&result->GPUInputOnDeviceTime, instart, instop);
        cudaEventSynchronize(outstop);
        cudaEventElapsedTime(&result->GPUOutputFromDeviceTime, outstart, outstop);
        result->blockGridInfo = blockGridInfo;
        result->GPUusedGlobalMemory = memoryUsed;
        result->GPUtotalGlobMemory = prop.totalGlobalMem;
        return;
    }

}
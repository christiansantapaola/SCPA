//
// Created by 9669c on 11/03/2022.
//


#include "CSRMatrix.h"
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
SpMV_CSR_kernel(int num_rows, const float *data, const int *col_index, const int *row_ptr, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int elem = row_start; elem < row_end; elem++) {
            dot += data[elem] * x[col_index[elem]];
        }
        y[row] += dot;
    }
}

SpMVResult CSRMatrix::SpMV_GPU(Vector &X, Vector &Y) {
    SpMVResult result = {false, 0, 0, 0,0};
    float *d_matrix_data, *d_x, *d_y;
    int *d_col_index, *d_row_ptr;
    cudaEvent_t start, stop, instart, instop, outstart, outstop;
    if (X.getSize() != col_size && Y.getSize() != row_size) {
        return result;
    }

    CudaDeviceInfo deviceInfo = CudaDeviceInfo();
    deviceInfo.setDevice(deviceInfo.getBestDevice());
    BlockGridInfo blockGridInfo = deviceInfo.getBlockSize(row_size);
    printf("CSR: %zu %zu\n", blockGridInfo.blockSize, blockGridInfo.gridSize);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&instart);
    cudaEventCreate(&instop);
    cudaEventCreate(&outstart);
    cudaEventCreate(&outstop);

    auto t0 = std::chrono::high_resolution_clock::now();

    cudaEventRecord(instart);

    checkCudaErrors(cudaMalloc(&d_matrix_data, num_non_zero_elements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_col_index, num_non_zero_elements * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_row_ptr, (row_size + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_y, row_size * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_matrix_data, data,num_non_zero_elements * sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, Y.getData(), row_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_col_index, col_index, num_non_zero_elements * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row_ptr, row_pointer, (row_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_x, row_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_x, X.getData(), row_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(instop);
    cudaEventRecord(start);
    SpMV_CSR_kernel<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(row_size,
                                d_matrix_data,
                                d_col_index,
                                d_row_ptr,
                                d_x,
                                d_y);
    cudaEventRecord(stop);

    cudaEventRecord(outstart);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaMemcpy(Y.getData(), d_y, row_size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_matrix_data));
    checkCudaErrors(cudaFree(d_col_index));
    checkCudaErrors(cudaFree(d_row_ptr));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    cudaEventSynchronize(outstop);
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.GPUKernelExecutionTime, start, stop);
    cudaEventSynchronize(instop);
    cudaEventElapsedTime(&result.GPUInputOnDeviceTime, instart, instop);
    cudaEventSynchronize(outstop);
    cudaEventElapsedTime(&result.GPUOutputFromDeviceTime, outstart, outstop);
    result.CPUFunctionExecutionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    result.success = true;
    return result;
}
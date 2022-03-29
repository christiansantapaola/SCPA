//
// Created by 9669c on 21/03/2022.
//

#include <cuda.h>

#include "cudaUtils.cuh"
#include "ELLMatrix.h"

__global__ void SpMV_ELL(int num_rows, const float *data, const int *col_index, int num_elem, const float *x, float *y) {
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

SpMVResult ELLMatrix::SpMV_GPU(Vector &X, Vector &Y) {
    SpMVResult result = {false, 0, 0, 0,0};
    float *d_x, *d_y;
    float *d_data;
    int *d_col_index;
    cudaEvent_t start, stop, instart, instop, outstart, outstop;
    if (X.getSize() != col_size && Y.getSize() != row_size) {
        result.success = false;
        return result;
    }
    CudaDeviceInfo deviceInfo = CudaDeviceInfo();
    deviceInfo.setDevice(deviceInfo.getBestDevice());
    size_t memory_used = X.getSize() * sizeof(float)  + Y.getSize() * sizeof(float) + data_size * sizeof(float) + data_size * sizeof(int);
    size_t memory_available = deviceInfo.getDeviceProp(deviceInfo.dev)->totalGlobalMem;
    printf("mem: %zu/%zu\n", memory_used, memory_available);
    if ( memory_used >= memory_available) {
        return result;
    }
    BlockGridInfo blockGridInfo = deviceInfo.getBlockSize(row_size);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&instart);
    cudaEventCreate(&instop);
    cudaEventCreate(&outstart);
    cudaEventCreate(&outstop);

    auto t0 = std::chrono::high_resolution_clock::now();

    cudaEventRecord(instart);
    checkCudaErrors(cudaMalloc(&(d_x), X.getSize() * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_y), Y.getSize() * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_data), data_size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_col_index), data_size * sizeof (int )));

    checkCudaErrors(cudaMemcpy(d_x, X.getData(), X.getSize() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, Y.getData(), Y.getSize() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_data, data, data_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_col_index, col_index, num_non_zero_elements * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(instop);
    cudaEventRecord(start);
    SpMV_ELL<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(row_size, d_data, d_col_index, num_elem, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventRecord(outstart);
    checkCudaErrors(cudaMemcpy(Y.getData(), d_y, Y.getSize() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_col_index));
    cudaEventRecord(outstop);
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
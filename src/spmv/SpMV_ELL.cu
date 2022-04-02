//
// Created by 9669c on 21/03/2022.
//

#include <cuda.h>

#include "cudaUtils.cuh"
extern "C" {
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}

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

void ELLMatrix_SpMV_GPU(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResult *result) {
    float *d_x, *d_y;
    float *d_data;
    int *d_col_index;
    cudaEvent_t start, stop, instart, instop, outstart, outstop;

    size_t memoryUsed;
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
    memoryUsed = (matrix->num_non_zero_elements + x->size + y->size) * sizeof(float) +   sizeof(int) * (matrix->row_size + 1 + matrix->num_non_zero_elements);
    int bestDev = CudaUtils_getBestDevice();
    CudaUtils_setDevice(bestDev);
    cudaDeviceProp prop;
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
    checkCudaErrors(cudaMalloc(&(d_x), x->size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_y), y->size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_data), matrix->data_size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_col_index), matrix->data_size * sizeof (int )));

    checkCudaErrors(cudaMemcpyAsync(d_x, x->data, x->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_y, y->data, y->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_data, matrix->data, matrix->data_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_col_index, matrix->col_index, matrix->num_non_zero_elements * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(instop);
    cudaEventRecord(start);
    SpMV_ELL<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(matrix->row_size, d_data, d_col_index, matrix->num_elem, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    checkCudaErrors(cudaPeekAtLastError());
    cudaEventRecord(outstart);
    checkCudaErrors(cudaMemcpy(y->data, d_y, y->size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_col_index));
    cudaEventRecord(outstop);
    cudaEventSynchronize(stop);
    if (result) {
        result->success = 1;
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
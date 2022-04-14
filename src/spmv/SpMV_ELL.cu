#include <cuda.h>

#include "cudaUtils.cuh"
extern "C" {
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}

__global__ void SpMV_ELL(u_int64_t num_rows, const float *data, const u_int64_t *col_index, u_int64_t num_elem, const float *x, float *y) {
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

void ELLMatrix_SpMV_GPU(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResultCUDA *result) {
    float *d_x, *d_y;
    float *d_data;
    u_int64_t *d_col_index;
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
    memoryUsed = (matrix->data_size + x->size + y->size) * sizeof(float) +   sizeof(u_int64_t) * (matrix->data_size);
    int bestDev = CudaUtils_getBestDevice(memoryUsed);
    if (bestDev == -1) {
        fprintf(stderr,"%s\n", "NOT ENOUGH MEMORY");
        exit(EXIT_FAILURE);
    }
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
    checkCudaErrors(cudaMalloc(&(d_col_index), matrix->data_size * sizeof (u_int64_t)));

//    checkCudaErrors(cudaMemcpyAsync(d_x, x->data, x->size * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_y, y->data, y->size * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_data, matrix->data, matrix->data_size * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_col_index, matrix->col_index, matrix->num_non_zero_elements * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(d_x, x->data, x->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y->data, y->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_data, matrix->data, matrix->data_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_col_index, matrix->col_index, matrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));
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
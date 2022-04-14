extern "C" {
#include "COOMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
}
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaUtils.cuh"



__global__ void
SpMV_COO_kernel(u_int64_t num_elements, const float *data, const u_int64_t *col_index, const u_int64_t *row_index, const float *x, float *y) {
    u_int64_t elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem < num_elements) {
        atomicAdd(&y[row_index[elem]], data[elem] * x[col_index[elem]]);
    }
}

extern "C"
void COOMatrix_SpMV_GPU(const COOMatrix *matrix, const Vector *x, Vector *y, SpMVResultCUDA *result) {
    float *d_matrix_data, *d_x, *d_y;
    u_int64_t *d_col_index, *d_row_index;
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
    memoryUsed = (matrix->num_non_zero_elements + x->size + y->size) * sizeof(float) +   sizeof(u_int64_t) * (2 * matrix->num_non_zero_elements);
    int bestDev = CudaUtils_getBestDevice(memoryUsed);
    if (bestDev == -1) {
        fprintf(stderr,"%s\n", "NOT ENOUGH MEMORY");
        exit(EXIT_FAILURE);
    }
    CudaUtils_setDevice(bestDev);
    cudaDeviceProp prop;
    BlockGridInfo blockGridInfo;
    CudaUtils_getDeviceProp(bestDev, &prop);
    CudaUtils_getBestCudaParameters(matrix->num_non_zero_elements, &prop, &blockGridInfo);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&instart);
    cudaEventCreate(&instop);
    cudaEventCreate(&outstart);
    cudaEventCreate(&outstop);


    cudaEventRecord(instart);

    checkCudaErrors(cudaMalloc(&d_matrix_data, matrix->num_non_zero_elements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_col_index, matrix->num_non_zero_elements * sizeof(u_int64_t)));
    checkCudaErrors(cudaMalloc(&d_row_index, matrix->num_non_zero_elements * sizeof(u_int64_t)));
    checkCudaErrors(cudaMalloc(&d_x, matrix->row_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y, matrix->row_size * sizeof(float)));

//    checkCudaErrors(cudaMemcpyAsync(d_matrix_data, matrix->data,matrix->num_non_zero_elements * sizeof(float),cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_col_index, matrix->col_index, matrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_row_index, matrix->row_index, (matrix->row_size + 1) * sizeof(u_int64_t), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_x, x->data, matrix->row_size * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpyAsync(d_y, y->data, matrix->row_size * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(d_matrix_data, matrix->data,matrix->num_non_zero_elements * sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_col_index, matrix->col_index, matrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row_index, matrix->row_index, matrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x->data, matrix->row_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y->data, matrix->row_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(instop);
    cudaEventRecord(start);
    SpMV_COO_kernel<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(matrix->num_non_zero_elements,
                                                                         d_matrix_data,
                                                                         d_col_index,
                                                                         d_row_index,
                                                                         d_x,
                                                                         d_y);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop);

    cudaEventRecord(outstart);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaMemcpy(y->data, d_y, matrix->row_size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_matrix_data));
    checkCudaErrors(cudaFree(d_col_index));
    checkCudaErrors(cudaFree(d_row_index));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
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
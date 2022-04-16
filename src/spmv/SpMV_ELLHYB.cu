#include <cuda.h>

#include "cudaUtils.cuh"
extern "C" {
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}

#include "SpMVKernel.cuh"

extern "C" void ELLMatrixHyb_SpMV_GPU(const ELLMatrix *ellMatrix, const COOMatrix *cooMatrix, const Vector *x, Vector *y, SpMVResultCUDA *result) {
    float *d_coomatrix_data, *d_ellmatrix_data, *d_x, *d_y;
    u_int64_t *d_coomatrix_col_index, *d_coomatrix_row_index, *d_ellmatrix_col_index;
    cudaEvent_t start, stop, instart, instop, outstart, outstop;
    cudaEvent_t cooStart, cooStop, cooInstart, cooInstop, cooOutstart, cooOutstop;
    size_t memoryUsed;
    cudaDeviceProp prop;
    BlockGridInfo cooBlockGridInfo, ellBlockGridInfo;
    if (!cooMatrix || !ellMatrix || !x || !y) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (x->size != cooMatrix->col_size && y->size != cooMatrix->row_size &&
        cooMatrix->row_size != ellMatrix->row_size && cooMatrix->col_size != cooMatrix->row_size) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (result) {
        memset(result, 0, sizeof(*result));
    }

    memoryUsed = (ellMatrix->data_size + x->size + y->size) * sizeof(float) + sizeof(u_int64_t) * (ellMatrix->data_size);
    int bestDev = CudaUtils_getBestDevice(memoryUsed);
    if (bestDev == -1) {
        fprintf(stderr,"%s\n", "NOT ENOUGH MEMORY");
        exit(EXIT_FAILURE);
    }
    CudaUtils_setDevice(bestDev);
    CudaUtils_getDeviceProp(bestDev, &prop);
    CudaUtils_getBestCudaParameters(ellMatrix->row_size, &prop, &ellBlockGridInfo);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&instart);
    cudaEventCreate(&instop);
    cudaEventCreate(&outstart);
    cudaEventCreate(&outstop);
    cudaEventCreate(&cooStart);
    cudaEventCreate(&cooStop);
    cudaEventCreate(&cooInstart);
    cudaEventCreate(&cooInstop);
    cudaEventCreate(&cooOutstart);
    cudaEventCreate(&cooOutstop);

    cudaEventRecord(instart);

    checkCudaErrors(cudaMalloc(&(d_x), x->size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_y), y->size * sizeof (float )));

    checkCudaErrors(cudaMalloc(&(d_ellmatrix_data), ellMatrix->data_size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_ellmatrix_col_index), ellMatrix->data_size * sizeof (u_int64_t)));

    checkCudaErrors(cudaMemcpy(d_x, x->data, x->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y->data, y->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ellmatrix_data, ellMatrix->data, ellMatrix->data_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ellmatrix_col_index, ellMatrix->col_index, ellMatrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));

    cudaEventRecord(instop);
    cudaEventRecord(start);
    SpMV_ELL_kernel<<<ellBlockGridInfo.gridSize, ellBlockGridInfo.blockSize>>>(ellMatrix->row_size, d_ellmatrix_data, d_ellmatrix_col_index, ellMatrix->num_elem, d_x, d_y);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    checkCudaErrors(cudaFree(d_ellmatrix_data));
    checkCudaErrors(cudaFree(d_ellmatrix_col_index));

    memoryUsed = (cooMatrix->num_non_zero_elements + x->size + y->size) * sizeof(float) + sizeof(u_int64_t) * (2 * cooMatrix->num_non_zero_elements);
    bestDev = CudaUtils_getBestDevice(memoryUsed);
    if (bestDev == -1) {
        fprintf(stderr,"%s\n", "NOT ENOUGH MEMORY");
        exit(EXIT_FAILURE);
    }
    CudaUtils_setDevice(bestDev);
    CudaUtils_getDeviceProp(bestDev, &prop);
    CudaUtils_getBestCudaParameters(cooMatrix->num_non_zero_elements, &prop, &cooBlockGridInfo);

    cudaEventRecord(cooInstart);

    checkCudaErrors(cudaMalloc(&d_coomatrix_data, cooMatrix->num_non_zero_elements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_coomatrix_col_index, cooMatrix->num_non_zero_elements * sizeof(u_int64_t)));
    checkCudaErrors(cudaMalloc(&d_coomatrix_row_index, cooMatrix->num_non_zero_elements * sizeof(u_int64_t)));

    checkCudaErrors(cudaMemcpy(d_coomatrix_data, cooMatrix->data, cooMatrix->num_non_zero_elements * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_coomatrix_col_index, cooMatrix->col_index, cooMatrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_coomatrix_row_index, cooMatrix->row_index, cooMatrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));

    cudaEventRecord(cooInstop);
    cudaEventRecord(cooStart);
    SpMV_COO_kernel<<<cooBlockGridInfo.gridSize, cooBlockGridInfo.blockSize>>>(cooMatrix->num_non_zero_elements,
                                                                               d_coomatrix_data,
                                                                               d_coomatrix_col_index,
                                                                               d_coomatrix_row_index,
                                                                               d_x,
                                                                               d_y);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(cooStop);

    cudaEventRecord(outstart);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaMemcpy(y->data, d_y, y->size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_coomatrix_data));
    checkCudaErrors(cudaFree(d_coomatrix_col_index));
    checkCudaErrors(cudaFree(d_coomatrix_row_index));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    cudaEventRecord(outstop);
    if (result) {
        float ellExTime, ellInTime, cooExTime, cooInTime;
        result->success = 1;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ellExTime, start, stop);
        cudaEventSynchronize(instop);
        cudaEventElapsedTime(&ellInTime, instart, instop);
        cudaEventSynchronize(cooStop);
        cudaEventElapsedTime(&cooExTime, cooStart, cooStop);
        cudaEventSynchronize(cooInstop);
        cudaEventElapsedTime(&cooInTime, cooInstart, cooInstop);
        cudaEventSynchronize(outstop);
        cudaEventElapsedTime(&result->GPUOutputFromDeviceTime, outstart, outstop);
        result->GPUKernelExecutionTime = ellExTime + cooExTime;
        result->GPUInputOnDeviceTime = ellInTime + cooInTime;
        result->GPUTotalTime = result->GPUInputOnDeviceTime + result->GPUKernelExecutionTime + result->GPUOutputFromDeviceTime;
        return;
    }
}

extern "C" void ELLMatrixHyb_SpMV_GPU_wpm(const ELLMatrix *ellMatrix, const COOMatrix *cooMatrix, const Vector *x, Vector *y, SpMVResultCUDA *result) {
    float *d_coomatrix_data, *d_ellmatrix_data, *d_x, *d_y;
    u_int64_t *d_coomatrix_col_index, *d_coomatrix_row_index, *d_ellmatrix_col_index;
    cudaEvent_t start, stop, instart, instop, outstart, outstop;
    cudaEvent_t cooStart, cooStop, cooInstart, cooInstop;
    size_t memoryUsed;
    cudaDeviceProp prop;
    BlockGridInfo cooBlockGridInfo, ellBlockGridInfo;
    SpMVResultCPU cooresult;
    if (!cooMatrix || !ellMatrix || !x || !y) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (x->size != cooMatrix->col_size && y->size != cooMatrix->row_size &&
        cooMatrix->row_size != ellMatrix->row_size && cooMatrix->col_size != cooMatrix->row_size) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (result) {
        memset(result, 0, sizeof(*result));
    }

    memoryUsed = (ellMatrix->data_size + x->size + y->size) * sizeof(float) + sizeof(u_int64_t) * (ellMatrix->data_size);
    int bestDev = CudaUtils_getBestDevice(memoryUsed);
    if (bestDev == -1) {
        fprintf(stderr,"%s\n", "NOT ENOUGH MEMORY");
        exit(EXIT_FAILURE);
    }
    CudaUtils_setDevice(bestDev);
    CudaUtils_getDeviceProp(bestDev, &prop);
    CudaUtils_getBestCudaParameters(ellMatrix->row_size, &prop, &ellBlockGridInfo);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&instart);
    cudaEventCreate(&instop);
    cudaEventCreate(&outstart);
    cudaEventCreate(&outstop);
    cudaEventCreate(&cooStart);
    cudaEventCreate(&cooStop);
    cudaEventCreate(&cooInstart);
    cudaEventCreate(&cooInstop);

    cudaEventRecord(instart);

    checkCudaErrors(cudaMalloc(&(d_x), x->size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_y), y->size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_ellmatrix_data), ellMatrix->data_size * sizeof (float )));
    checkCudaErrors(cudaMalloc(&(d_ellmatrix_col_index), ellMatrix->data_size * sizeof (u_int64_t)));
    checkCudaErrors(cudaMemcpyAsync(d_x, x->data, x->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_y, y->data, y->size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_ellmatrix_data, ellMatrix->data, ellMatrix->data_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_ellmatrix_col_index, ellMatrix->col_index, ellMatrix->num_non_zero_elements * sizeof(u_int64_t), cudaMemcpyHostToDevice));
    cudaEventRecord(instop);
    cudaEventRecord(start);
    SpMV_ELL_kernel<<<ellBlockGridInfo.gridSize, ellBlockGridInfo.blockSize>>>(ellMatrix->row_size, d_ellmatrix_data, d_ellmatrix_col_index, ellMatrix->num_elem, d_x, d_y);
    cudaEventRecord(stop);
    COOMatrix_SpMV_CPU(cooMatrix, x, y, &cooresult);
    cudaEventRecord(outstart);
    checkCudaErrors(cudaMemcpy(y->data, d_y, y->size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_ellmatrix_data));
    checkCudaErrors(cudaFree(d_ellmatrix_col_index));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    cudaEventRecord(outstop);
    if (result) {
        float ellExTime, ellInTime, cooExTime, cooInTime;
        result->success = 1;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ellExTime, start, stop);
        cudaEventSynchronize(instop);
        cudaEventElapsedTime(&ellInTime, instart, instop);
        cudaEventSynchronize(cooStop);
        cudaEventElapsedTime(&cooExTime, cooStart, cooStop);
        cudaEventSynchronize(cooInstop);
        cudaEventElapsedTime(&cooInTime, cooInstart, cooInstop);
        cudaEventSynchronize(outstop);
        cudaEventElapsedTime(&result->GPUOutputFromDeviceTime, outstart, outstop);
        result->GPUKernelExecutionTime = ellExTime;
        result->GPUInputOnDeviceTime = ellInTime;
        cudaEventElapsedTime(&result->GPUTotalTime, instart, outstop);
        result->CPUTime = cooresult.timeElapsed;
        return;
    }
}


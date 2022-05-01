#include <cuda.h>

#include "cudaUtils.cuh"
extern "C" {
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}

#include "SpMVKernel.cuh"

extern "C" int ELLCOOMatrix_SpMV_CUDA(const ELLMatrix *d_ellMatrix, const COOMatrix *h_cooMatrix, const Vector *h_x, Vector *h_y, SpMVResultCUDA *result) {
    cudaEvent_t start, stop;
    cudaDeviceProp prop;
    BlockGridInfo ellBlockGridInfo;
    SpMVResultCPU cooresult;
    if (!h_cooMatrix || !d_ellMatrix || !h_x || !h_y) {
        return SPMV_FAIL;
    }
    if (h_x->size != h_cooMatrix->col_size && h_y->size != h_cooMatrix->row_size &&
        h_cooMatrix->row_size != d_ellMatrix->row_size && h_cooMatrix->col_size != h_cooMatrix->row_size) {
        if (result) {
        }
        return SPMV_FAIL;
    }
    if (result) {
        memset(result, 0, sizeof(*result));
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    size_t memory_used = (d_ellMatrix->data_size + h_x->size + h_y->size) * sizeof(float) +   sizeof(u_int64_t) * (d_ellMatrix->data_size);
    int dev = CudaUtils_getBestDevice(memory_used);
    CudaUtils_setDevice(dev);
    CudaUtils_getDeviceProp(dev, &prop);
    CudaUtils_getBestCudaParameters(d_ellMatrix->row_size, &prop, &ellBlockGridInfo);
    Vector* d_x = Vector_to_CUDA(h_x);
    Vector* d_y = Vector_to_CUDA(h_y);
    cudaEventRecord(start);
    SpMV_ELL_kernel<<<ellBlockGridInfo.gridSize, ellBlockGridInfo.blockSize>>>(d_ellMatrix->row_size, d_ellMatrix->data, d_ellMatrix->col_index, d_ellMatrix->num_elem, d_x->data, d_y->data);
    cudaEventRecord(stop);
    COOMatrix_SpMV(h_cooMatrix, h_x, h_y, &cooresult);
    Vector *ellVector = Vector_from_CUDA(d_y);
    Vector_free_CUDA(d_x);
    Vector_free_CUDA(d_y);
    Vector_sum(h_y, ellVector);
    Vector_free(ellVector);
    if (result) {
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result->GPUKernelExecutionTime, start, stop);
        result->CPUTime = cooresult.timeElapsed;
    }
    return SPMV_SUCCESS;

}
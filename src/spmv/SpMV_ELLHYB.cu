#include <cuda.h>

#include "cudaUtils.cuh"
extern "C" {
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}

#include "SpMVKernel.cuh"

extern "C" int ELLCOOMatrix_SpMV_CUDA(int cudaDevice, const ELLMatrix *d_ellMatrix, const COOMatrix *h_cooMatrix, const Vector *h_x, Vector *h_y, float *time) {
    cudaEvent_t start, stop;
    cudaDeviceProp prop;
    BlockGridInfo BlockGridInfo;
    if (!h_cooMatrix || !d_ellMatrix || !h_x || !h_y) {
        return SPMV_FAIL;
    }
    if (h_x->size != h_cooMatrix->col_size && h_y->size != h_cooMatrix->row_size &&
        h_cooMatrix->row_size != d_ellMatrix->row_size && h_cooMatrix->col_size != h_cooMatrix->row_size) {
        return SPMV_FAIL;
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CudaUtils_getDeviceProp(cudaDevice, &prop);
    CudaUtils_getBestCudaParameters(d_ellMatrix->row_size, &prop, &BlockGridInfo);
    Vector* d_x = Vector_to_CUDA(h_x);
    Vector* d_y = Vector_to_CUDA(h_y);
    cudaEventRecord(start);
    SpMV_ELL_kernel<<<BlockGridInfo.gridSize, BlockGridInfo.blockSize>>>(d_ellMatrix->row_size, d_ellMatrix->data, d_ellMatrix->col_index, d_ellMatrix->num_elem, d_x->data, d_y->data);
    COOMatrix_SpMV(h_cooMatrix, h_x, h_y, NULL);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    Vector *h_ellY = Vector_from_CUDA(d_y);
    Vector_sum(h_y, h_ellY);
    if (time) {
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(time, start, stop);
        *time = *time / 1000.0;
    }
    Vector_free(h_ellY);
    Vector_free_CUDA(d_y);
    Vector_free_CUDA(d_x);
    return SPMV_SUCCESS;
}
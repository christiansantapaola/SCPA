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

int ELLMatrix_SpMV_CUDA(int cudaDevice, const ELLMatrix *d_matrix, const Vector *h_x, Vector *h_y) {
    //cudaEvent_t start, stop;
    Vector *d_x, *d_y;
    cudaDeviceProp prop;
    BlockGridInfo blockGridInfo;
    if (!d_matrix || !h_x || !h_y) {
        return SPMV_FAIL;
    }
    if (h_x->size != d_matrix->col_size && h_y->size != d_matrix->row_size) {
        return SPMV_FAIL;
    }
    CudaUtils_getDeviceProp(cudaDevice, &prop);
    CudaUtils_getBestCudaParameters(d_matrix->row_size, &prop, &blockGridInfo);
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    d_x = Vector_to_CUDA(h_x);
    d_y = Vector_to_CUDA(h_y);
    //cudaEventRecord(start);
    SpMV_ELL_kernel<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(d_matrix->row_size, d_matrix->data, d_matrix->col_index, d_matrix->num_elem, d_x->data, d_y->data);
    //cudaEventRecord(stop);
    Vector_copy_from_CUDA(h_y, d_y);
    Vector_free_CUDA(d_y);
    Vector_free_CUDA(d_x);
    return SPMV_SUCCESS;
}
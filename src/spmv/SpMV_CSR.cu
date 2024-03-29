extern "C" {
#include "CSRMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"
}
#include "cudaUtils.cuh"
#include <cuda.h>
#include "SpMVKernel.cuh"

#define MAX_X_SIZE 65536 / sizeof(float)

__global__ void
SpMV_CSR_kernel(u_int64_t num_rows, const float *data, const u_int64_t *col_index, const u_int64_t *row_ptr, const float *x, float *y) {
    const u_int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        const u_int64_t row_start = row_ptr[row];
        const u_int64_t row_end = row_ptr[row + 1];
        for (u_int64_t elem = row_start; elem < row_end; elem++) {
            dot += data[elem] * x[col_index[elem]];
        }
        y[row] += dot;
    }
}

extern "C" int CSRMatrix_SpMV_CUDA(int cudaDevice, const CSRMatrix *d_matrix, const Vector *h_x, Vector *h_y, float *time) {
    cudaEvent_t start, stop;
    Vector *d_x, *d_y;
    cudaDeviceProp prop;
    BlockGridInfo blockGridInfo;
    int minGridSize; // The minimum grid size needed to achieve the 
    if (!d_matrix || !h_x || !h_y) {
        return SPMV_FAIL;
    }
    if (h_x->size != d_matrix->col_size && h_y->size != d_matrix->row_size) {
        return SPMV_FAIL;
    }
    CudaUtils_getDeviceProp(cudaDevice, &prop);
    // CudaUtils_getBestCudaParameters(d_matrix->row_size, &prop, &blockGridInfo);
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, (int*)&blockGridInfo.blockSize, 
                                      SpMV_CSR_kernel, 0, 0); 
    // Round up according to array size 
    blockGridInfo.gridSize = (d_matrix->row_size + blockGridInfo.blockSize - 1) / blockGridInfo.blockSize; 

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    d_x = Vector_to_CUDA_async(h_x);
    d_y = Vector_to_CUDA_async(h_y);
    cudaEventRecord(start);
    SpMV_CSR_kernel<<<blockGridInfo.gridSize, blockGridInfo.blockSize>>>(d_matrix->row_size,
                                                                         d_matrix->data,
                                                                         d_matrix->col_index,
                                                                         d_matrix->row_pointer,
                                                                         d_x->data,
                                                                         d_y->data);
    cudaEventRecord(stop);
    Vector_copy_from_CUDA(h_y, d_y);
    if (time) {
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(time, start, stop);
        *time = *time / 1000.0; // convert ms -> s
    }
    Vector_free_CUDA(d_y);
    Vector_free_CUDA(d_x);
    return SPMV_SUCCESS;
}
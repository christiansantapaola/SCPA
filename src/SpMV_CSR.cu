//
// Created by 9669c on 11/03/2022.
//

#include "CSRMatrix.h"
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void SpMV_CSR_kernel(int num_rows, float *data, int *col_index, int *row_ptr, float *x, float *y);

__global__ void SpMV_CSR_kernel(int num_rows, float *data, int *col_index, int *row_ptr, float *x, float *y) {
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

void CSRMatrix::SpMV_GPU(float *h_x, float *h_y) {
    if (!h_x || !h_y) return;
    float *d_matrix_data, *d_x, *d_y;
    int *d_col_index, *d_row_ptr;
    gpuErrchk(cudaMalloc(&d_matrix_data,num_non_zero_elements * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_col_index, num_non_zero_elements * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_row_ptr, (row_size + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_x, row_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_y, row_size * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_matrix_data, data, num_non_zero_elements * sizeof(float),
                       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_x, h_x, row_size * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, h_y, row_size * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_index, col_index, num_non_zero_elements * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ptr, row_pointer, (row_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

    //dim3 grid(row_size, col_size, 1);
    //dim3 block(512, 512, 1);

    SpMV_CSR_kernel<<<4, 512>>>(row_size,
                                d_matrix_data,
                                d_col_index,
                                d_row_ptr,
                                d_x,
                                d_y);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_y, d_y, row_size * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_matrix_data));
    gpuErrchk(cudaFree(d_col_index));
    gpuErrchk(cudaFree(d_row_ptr));
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_y));
}

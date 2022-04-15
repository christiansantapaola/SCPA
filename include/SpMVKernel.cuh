//
// Created by 9669c on 14/04/2022.
//

#ifndef SPARSEMATRIX_SPMVKERNEL_CUH
#define SPARSEMATRIX_SPMVKERNEL_CUH

#include <cuda.h>
extern "C" {
#include "stdlib.h"
};

__global__ void SpMV_COO_kernel(u_int64_t num_elements, const float *data, const u_int64_t *col_index, const u_int64_t *row_index, const float *x, float *y);
__global__ void SpMV_CSR_kernel(u_int64_t num_rows, const float *data, const u_int64_t *col_index, const u_int64_t *row_ptr, const float *x, float *y);
__global__ void SpMV_ELL_kernel(u_int64_t num_rows, const float *data, const u_int64_t *col_index, u_int64_t num_elem, const float *x, float *y);

#endif //SPARSEMATRIX_SPMVKERNEL_CUH

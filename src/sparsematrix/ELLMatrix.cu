extern "C" {
#include "CSRMatrix.h"
#include "ELLMatrix.h"
}

#include <cuda.h>
#include "cudaUtils.cuh"

extern "C" ELLMatrix *ELLMatrix_pinned_memory_new(CSRMatrix *csrMatrix) {
    if (!csrMatrix) return NULL;
    ELLMatrix *ellMatrix = (ELLMatrix *) malloc(sizeof(ELLMatrix));
    ellMatrix->row_size = csrMatrix->row_size;
    ellMatrix->col_size = csrMatrix->col_size;
    ellMatrix->num_non_zero_elements = csrMatrix->num_non_zero_elements;

    // find the the maximum number of non zero elements in a row.
    int max_num_nz_elem = 0;
    for (int row = 0; row < csrMatrix->row_size; row++) {
        int num_nz_elem_curr_row = csrMatrix->row_pointer[row + 1] - csrMatrix->row_pointer[row];
        if (max_num_nz_elem < num_nz_elem_curr_row) {
            max_num_nz_elem = num_nz_elem_curr_row;
        }
    }

    ellMatrix->num_elem = max_num_nz_elem;
    ellMatrix->data_size = ellMatrix->row_size * ellMatrix->num_elem;

    checkCudaErrors(cudaHostAlloc(&ellMatrix->data, ellMatrix->data_size * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&ellMatrix->col_index, ellMatrix->data_size * sizeof(u_int64_t), cudaHostAllocDefault));
    // add padding;
    memset(ellMatrix->data, 0, ellMatrix->data_size * sizeof(float));
    memset(ellMatrix->col_index, 0, ellMatrix->data_size * sizeof(u_int64_t));
    for (int row = 0; row < ellMatrix->row_size; row++) {
        int row_start = csrMatrix->row_pointer[row];
        int num_nz_elem = csrMatrix->row_pointer[row + 1] - row_start;
        for (int i = 0; i < num_nz_elem; i++) {
            int index = row * ellMatrix->num_elem + i;
            ellMatrix->data[index] = csrMatrix->data[row_start + i];
            ellMatrix->col_index[index] = csrMatrix->col_index[row_start + i];
        }
    }
    return ellMatrix;
}
extern "C" void ELLMatrix_pinned_memory_free(ELLMatrix *ellMatrix) {
    if (!ellMatrix) return;
    checkCudaErrors(cudaFreeHost(ellMatrix->data));
    checkCudaErrors(cudaFreeHost(ellMatrix->col_index));
    free(ellMatrix);
}

extern "C" {
#include "COOMatrix.h"
#include <stdlib.h>
}
#include "cudaUtils.cuh"
#include <cuda_runtime.h>

extern "C" int COOMatrix_split_wpm(const COOMatrix *matrix, COOMatrix *first, COOMatrix *second, u_int64_t threshold) {
    if (!matrix || !first || !second) return -1;
    Histogram *rowsElem = Histogram_new(matrix->row_size + 1);
    first->row_size = matrix->row_size;
    first->col_size = matrix->col_size;
    first->num_non_zero_elements = 0;
    second->row_size = matrix->row_size;
    second->col_size = matrix->col_size;
    second->num_non_zero_elements = 0;
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
        Histogram_insert(rowsElem, matrix->row_index[i]);
    }
    for (u_int64_t i = 0; i < matrix->row_size; i++) {
        u_int64_t numElem = Histogram_getElemAtIndex(rowsElem, i);
        if (numElem <= threshold) {
            first->num_non_zero_elements += numElem;
        } else {
            second->num_non_zero_elements += numElem;
        }
    }
    if (first->num_non_zero_elements == 0 || second->num_non_zero_elements == 0) {
        Histogram_free(rowsElem);
        return 1;
    }
    first->row_index = (u_int64_t *)malloc(first->num_non_zero_elements * sizeof(u_int64_t));
    first->col_index = (u_int64_t *)malloc(first->num_non_zero_elements * sizeof(u_int64_t));
    first->data = (float *)malloc(first->num_non_zero_elements * sizeof(float));
    memset(first->data, 0,first->num_non_zero_elements * sizeof(float ) );
    memset(first->row_index, 0,first->num_non_zero_elements * sizeof(u_int64_t) );
    memset(first->col_index, 0,first->num_non_zero_elements * sizeof(u_int64_t) );



    checkCudaErrors(cudaHostAlloc(&second->row_index, second->num_non_zero_elements * sizeof(u_int64_t), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&second->col_index, second->num_non_zero_elements * sizeof(u_int64_t), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&second->data, second->num_non_zero_elements * sizeof(float), cudaHostAllocDefault));

    memset(second->data, 0,second->num_non_zero_elements * sizeof(float ) );
    memset(second->row_index, 0,second->num_non_zero_elements * sizeof(u_int64_t) );
    memset(second->col_index, 0,second->num_non_zero_elements * sizeof(u_int64_t) );


    u_int64_t fpos = 0, spos = 0;
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
        u_int64_t numElem = Histogram_getElemAtIndex(rowsElem, matrix->row_index[i]);
        if (numElem <= threshold) {
            first->row_index[fpos] = matrix->row_index[i];
            first->col_index[fpos] = matrix->col_index[i];
            first->data[fpos] = matrix->data[i];
            fpos++;
        } else {
            second->row_index[spos] = matrix->row_index[i];
            second->col_index[spos] = matrix->col_index[i];
            second->data[spos] = matrix->data[i];
            spos++;
        }
    }
    Histogram_free(rowsElem);
    return 0;
}

extern "C" void COOMatrix_free_wpm(COOMatrix *matrix) {
    if (!matrix) return;
    cudaFreeHost(matrix->data);
    cudaFreeHost(matrix->col_index);
    cudaFreeHost(matrix->row_index);
    free(matrix);
}

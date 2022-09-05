extern "C" {
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "util.h"
}

#include <cuda.h>
#include "cudaUtils.cuh"

extern "C" ELLMatrix *ELLMatrix_new_fromCOO_wpm(const COOMatrix *cooMatrix) {
    if (!cooMatrix) return NULL;
    ELLMatrix *ellMatrix = (ELLMatrix *) malloc(sizeof(ELLMatrix));
    ellMatrix->row_size = cooMatrix->row_size;
    ellMatrix->col_size = cooMatrix->col_size;
    ellMatrix->num_non_zero_elements = cooMatrix->num_non_zero_elements;

    // find the the maximum number of non zero elements in a row.
    ellMatrix->num_elem = COOMatrix_maxNumberOfElem(cooMatrix);
    ellMatrix->data_row_size = ellMatrix->row_size;
    ellMatrix->data_col_size = ellMatrix->num_elem;
    ellMatrix->data_size = ellMatrix->row_size * ellMatrix->num_elem;

    checkCudaErrors(cudaHostAlloc(&ellMatrix->data, ellMatrix->data_size * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&ellMatrix->col_index, ellMatrix->data_size * sizeof(u_int64_t), cudaHostAllocDefault));
    // add padding;
    memset(ellMatrix->data, 0, ellMatrix->data_size * sizeof(float));
    memset(ellMatrix->col_index, 0, ellMatrix->data_size * sizeof(u_int64_t));
    Histogram *elemInserted = Histogram_new(cooMatrix->row_size + 1);
    for (u_int64_t i = 0; i < cooMatrix->num_non_zero_elements; i++) {
        u_int64_t row = cooMatrix->row_index[i];
        u_int64_t col = cooMatrix->col_index[i];
        float data = cooMatrix->data[i];
        u_int64_t base = row * ellMatrix->num_elem;
        u_int64_t offset = Histogram_getElemAtIndex(elemInserted, row);
        ellMatrix->data[base + offset] = data;
        ellMatrix->col_index[base + offset] = col;
        Histogram_insert(elemInserted, row);
    }
    Histogram_free(elemInserted);
    return ellMatrix;
}

extern "C" ELLMatrix *ELLMatrix_new_fromCSR_wpm(const CSRMatrix *csrMatrix) {
    if (!csrMatrix) return NULL;
    ELLMatrix *ellMatrix = (ELLMatrix *) malloc(sizeof(ELLMatrix));
    ellMatrix->row_size = csrMatrix->row_size;
    ellMatrix->col_size = csrMatrix->col_size;
    ellMatrix->num_non_zero_elements = csrMatrix->num_non_zero_elements;

    // find the the maximum number of non zero elements in a row.
    u_int64_t max_num_nz_elem = 0;
    for (u_int64_t row = 0; row < csrMatrix->row_size; row++) {
        u_int64_t num_nz_elem_curr_row = csrMatrix->row_pointer[row + 1] - csrMatrix->row_pointer[row];
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
    for (u_int64_t row = 0; row < ellMatrix->row_size; row++) {
        u_int64_t row_start = csrMatrix->row_pointer[row];
        u_int64_t num_nz_elem = csrMatrix->row_pointer[row + 1] - row_start;
        for (u_int64_t i = 0; i < num_nz_elem; i++) {
            u_int64_t index = row * ellMatrix->num_elem + i;
            ellMatrix->data[index] = csrMatrix->data[row_start + i];
            ellMatrix->col_index[index] = csrMatrix->col_index[row_start + i];
        }
    }
    return ellMatrix;
}

extern "C" void ELLMatrix_free_wpm(ELLMatrix *ellMatrix) {
    if (!ellMatrix) return;
    checkCudaErrors(cudaFreeHost(ellMatrix->data));
    checkCudaErrors(cudaFreeHost(ellMatrix->col_index));
    free(ellMatrix);
}

extern "C" void ELLMatrix_transpose_wpm(const ELLMatrix *ellMatrix, ELLMatrix *transposed) {
    if (!transposed) {
        return;
    }
    transposed->col_size = ellMatrix->row_size;
    transposed->row_size = ellMatrix->col_size;
    transposed->data_col_size = ellMatrix->data_row_size;
    transposed->data_row_size = ellMatrix->data_col_size;
    transposed->data_size = ellMatrix->data_size;
    transposed->num_elem = ellMatrix->num_elem;
    transposed->num_non_zero_elements = ellMatrix->num_non_zero_elements;
    checkCudaErrors(cudaHostAlloc(&transposed->data, ellMatrix->data_size * sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&transposed->col_index, ellMatrix->data_size * sizeof(u_int64_t), cudaHostAllocDefault));
    //memcpy(transposed->data, ellMatrix->data, ellMatrix->data_size * sizeof(float));
    //memcpy(transposed->col_index ellMatrix->col_index, ellMatrix->data_size * sizeof(u_int64_t));
    transposef(ellMatrix->data, transposed->data, ellMatrix->data_row_size, ellMatrix->data_col_size);
    transpose_u_int64_t(ellMatrix->col_index, transposed->col_index, ellMatrix->data_row_size, ellMatrix->data_col_size);
}


extern "C" ELLMatrix *ELLMatrix_to_CUDA(const ELLMatrix *h_matrix) {
    if (!h_matrix) {
        return NULL;
    }
    ELLMatrix *h_transposed_matrix = (ELLMatrix *)malloc(sizeof(ELLMatrix));
    ELLMatrix_transpose_wpm(h_matrix, h_transposed_matrix);
    ELLMatrix *d_matrix = (ELLMatrix *)malloc(sizeof(ELLMatrix));
    d_matrix->col_size = h_transposed_matrix->row_size;
    d_matrix->row_size = h_transposed_matrix->col_size;
    d_matrix->data_col_size = h_transposed_matrix->data_row_size;
    d_matrix->data_row_size = h_transposed_matrix->data_col_size;
    d_matrix->data_size = h_transposed_matrix->data_size;
    d_matrix->num_elem = h_transposed_matrix->num_elem;
    d_matrix->num_non_zero_elements = h_transposed_matrix->num_non_zero_elements;
    checkCudaErrors(cudaMalloc(&d_matrix->data, d_matrix->data_size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_matrix->col_index, d_matrix->data_size * sizeof(u_int64_t)));
    checkCudaErrors(cudaMemcpyAsync(d_matrix->data, h_transposed_matrix->data, h_transposed_matrix->data_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_matrix->col_index, h_transposed_matrix->col_index, h_transposed_matrix->data_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    ELLMatrix_free_wpm(h_transposed_matrix);
    return d_matrix;
}

extern "C" void ELLMatrix_free_CUDA(ELLMatrix *d_matrix) {
    if (!d_matrix) return;
    checkCudaErrors(cudaFree(d_matrix->data));
    checkCudaErrors(cudaFree(d_matrix->col_index));
    free(d_matrix);
}

extern "C" {
#include "CSRMatrix.h"
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaUtils.cuh"

extern "C" CSRMatrix *CSRMatrix_new_wpm(const COOMatrix *cooMatrix) {
    if (!cooMatrix) return NULL;
    CSRMatrix *csrMatrix = NULL;
    csrMatrix = (CSRMatrix *) malloc(sizeof(CSRMatrix));
    csrMatrix->row_size = cooMatrix->row_size;
    csrMatrix->col_size = cooMatrix->col_size;
    csrMatrix->num_non_zero_elements = cooMatrix->num_non_zero_elements;
    checkCudaErrors(cudaHostAlloc(&csrMatrix->data, csrMatrix->num_non_zero_elements * sizeof(float ), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&csrMatrix->col_index, csrMatrix->num_non_zero_elements * sizeof(u_int64_t), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&csrMatrix->row_pointer, (csrMatrix->row_size + 1) * sizeof (u_int64_t), cudaHostAllocDefault));
    Histogram *elemForRow = Histogram_new(csrMatrix->row_size);

    // mi calcolo prima la posizione in base alle righe, poi aggiungo il resto,
    // questo perchè gli elementi in COO non devono essere ordinati.
    for (u_int64_t i = 0; i < csrMatrix->num_non_zero_elements; i++) {
        Histogram_insert(elemForRow, cooMatrix->row_index[i]);
    }
    u_int64_t count = 0;
    for (u_int64_t i = 0; i < cooMatrix->row_size + 1; i++) {
        csrMatrix->row_pointer[i] = count;
        count += Histogram_getElemAtIndex(elemForRow, i);
    }
    /*
     * Qui uso un istogramma per salvarmi il numero di inserimenti alla riga i.
     */
    Histogram *elemInsertedForRow = Histogram_new(csrMatrix->row_size);
    for (u_int64_t i = 0; i < csrMatrix->num_non_zero_elements; i++) {
        u_int64_t row = cooMatrix->row_index[i];
        u_int64_t col = cooMatrix->col_index[i];
        float val = cooMatrix->data[i];
        int64_t offset = Histogram_getElemAtIndex(elemInsertedForRow, row);
        u_int64_t index = csrMatrix->row_pointer[row] + offset;
        csrMatrix->data[index] = val;
        csrMatrix->col_index[index] = col;
        Histogram_insert(elemInsertedForRow, row);
    }
    Histogram_free(elemForRow);
    Histogram_free(elemInsertedForRow);
    return csrMatrix;
}

extern "C" void CSRMatrix_free_wpm(CSRMatrix *csrMatrix) {
    if (!csrMatrix) return;
    checkCudaErrors(cudaFreeHost(csrMatrix->data));
    checkCudaErrors(cudaFreeHost(csrMatrix->col_index));
    checkCudaErrors(cudaFreeHost(csrMatrix->row_pointer));
    free(csrMatrix);
}

extern "C" CSRMatrix *CSRMatrix_to_CUDA(const CSRMatrix *h_matrix) {
    if (!h_matrix) {
        return NULL;
    }
    CSRMatrix *d_matrix = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    if (!d_matrix) {
        return NULL;
    }
    d_matrix->row_size = h_matrix->row_size;
    d_matrix->col_size = h_matrix->col_size;
    d_matrix->num_non_zero_elements = h_matrix->num_non_zero_elements;
    checkCudaErrors(cudaMalloc(&d_matrix->data, sizeof(float) * d_matrix->num_non_zero_elements));
    checkCudaErrors(cudaMalloc(&d_matrix->col_index, sizeof(u_int64_t) * d_matrix->num_non_zero_elements));
    checkCudaErrors(cudaMalloc(&d_matrix->row_pointer, sizeof(u_int64_t) * (d_matrix->row_size + 1)));
    checkCudaErrors(cudaMemcpyAsync(d_matrix->data, h_matrix->data, sizeof(float) * d_matrix->num_non_zero_elements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_matrix->col_index, h_matrix->col_index, sizeof(u_int64_t) * d_matrix->num_non_zero_elements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_matrix->row_pointer, h_matrix->row_pointer, sizeof(u_int64_t) * (d_matrix->row_size + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    return d_matrix;
}

extern "C" void CSRMatrix_free_CUDA(CSRMatrix *d_matrix) {
    if (!d_matrix) return;
    checkCudaErrors(cudaFree(d_matrix->row_pointer));
    checkCudaErrors(cudaFree(d_matrix->col_index));
    checkCudaErrors(cudaFree(d_matrix->data));
    free(d_matrix);
}
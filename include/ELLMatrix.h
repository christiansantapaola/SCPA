#ifndef SPARSEMATRIX_ELLMATRIX_H
#define SPARSEMATRIX_ELLMATRIX_H

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "util.h"
#include <stdio.h>
#include <memory.h>

typedef struct ELLMatrix {
    float *data;
    u_int64_t *col_index;
    size_t data_size;
    u_int64_t data_row_size;
    u_int64_t data_col_size;
    u_int64_t num_elem;
    u_int64_t row_size;
    u_int64_t col_size;
    u_int64_t num_non_zero_elements;
} ELLMatrix;

ELLMatrix *ELLMatrix_new(const CSRMatrix *csrMatrix);
ELLMatrix *ELLMatrix_new_fromCOO(const COOMatrix *cooMatrix);
void ELLMatrix_free(ELLMatrix *ellMatrix);
ELLMatrix *ELLMatrix_new_fromCSR_wpm(const CSRMatrix *csrMatrix);
ELLMatrix *ELLMatrix_new_fromCOO_wpm(const COOMatrix *cooMatrix);
ELLMatrix *ELLMatrix_to_CUDA(const ELLMatrix *h_matrix);
void ELLMatrix_free_CUDA(ELLMatrix *d_ellMatrix);

void ELLMatrix_free_wpm(ELLMatrix *ellMatrix);
void ELLMatrix_transpose(const ELLMatrix *ellMatrix, ELLMatrix *transposed);
void ELLMatrix_outAsJSON(ELLMatrix *matrix, FILE *out);
void ELLMatrix_infoOutAsJSON(ELLMatrix *matrix, FILE *out);


#endif //SPARSEMATRIX_ELLMATRIX_H

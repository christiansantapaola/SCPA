#ifndef SPARSEMATRIX_ELLMATRIX_H
#define SPARSEMATRIX_ELLMATRIX_H

#include "COOMatrix.h"
#include "CSRMatrix.h"
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

ELLMatrix *ELLMatrix_new(CSRMatrix *csrMatrix);
ELLMatrix *ELLMatrix_new_fromCOO(COOMatrix *cooMatrix);
void ELLMatrix_free(ELLMatrix *ellMatrix);
ELLMatrix *ELLMatrix_new_wpm(CSRMatrix *csrMatrix);
ELLMatrix *ELLMatrix_new_fromCOO_wpm(COOMatrix *cooMatrix);
void ELLMatrix_free_wpm(ELLMatrix *ellMatrix);
void ELLMatrix_transpose(ELLMatrix *ellMatrix);
void ELLMatrix_outAsJSON(ELLMatrix *matrix, FILE *out);
void ELLMatrix_infoOutAsJSON(ELLMatrix *matrix, FILE *out);


#endif //SPARSEMATRIX_ELLMATRIX_H

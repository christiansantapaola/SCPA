//
// Created by 9669c on 14/03/2022.
//

#ifndef SPARSEMATRIX_CSRMATRIX_H
#define SPARSEMATRIX_CSRMATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include "COOMatrix.h"
#include "Histogram.h"
#include "mmio.h"

typedef struct CSRMatrix {
    float *data;
    u_int64_t *col_index;
    u_int64_t *row_pointer;
    u_int64_t num_non_zero_elements;
    u_int64_t row_size;
    u_int64_t col_size;
} CSRMatrix;

CSRMatrix *CSRMatrix_new(COOMatrix *matrix);
void CSRMatrix_free(CSRMatrix *matrix);
CSRMatrix *CSRMatrix_pinned_memory_new(COOMatrix *matrix);
void CSRMatrix_pinned_memory_free(CSRMatrix *matrix);
void CSRMatrix_outAsJSON(CSRMatrix *matrix, FILE *out);
void CSRMatrix_infoOutAsJSON(CSRMatrix *matrix, FILE *out);


#endif //SPARSEMATRIX_CSRMATRIX_H

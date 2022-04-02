//
// Created by 9669c on 14/03/2022.
//

#ifndef SPARSEMATRIX_CSRMATRIX_H
#define SPARSEMATRIX_CSRMATRIX_H

#include <stdio.h>
#include "COOMatrix.h"
#include "Histogram.h"
#include "mmio.h"

typedef struct CSRMatrix {
    float *data;
    int *col_index;
    int *row_pointer;
    int num_non_zero_elements;
    int row_size;
    int col_size;
} CSRMatrix;

CSRMatrix *CSRMatrix_new(COOMatrix *matrix);
void CSRMatrix_free(CSRMatrix *matrix);
CSRMatrix *CSRMatrix_pinned_memory_new(COOMatrix *matrix);
void CSRMatrix_pinned_memory_free(CSRMatrix *matrix);
void CSRMatrix_outAsJSON(CSRMatrix *matrix, FILE *out);


#endif //SPARSEMATRIX_CSRMATRIX_H

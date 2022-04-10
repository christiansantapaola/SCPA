//
// Created by 9669c on 14/03/2022.
//

#ifndef SPARSEMATRIX_COOMATRIX_H
#define SPARSEMATRIX_COOMATRIX_H


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mmio.h"

typedef struct COOMatrix {
    float *data;
    u_int64_t *col_index;
    u_int64_t *row_index;
    u_int64_t row_size;
    u_int64_t col_size;
    u_int64_t num_non_zero_elements;
} COOMatrix;

// COOMatrix *COOMatrix_new(const float *Matrix, int rows, int cols);
COOMatrix *COOMatrix_new(FILE *f);
void COOMatrix_free(COOMatrix *matrix);
void COOMatrix_outAsJSON(COOMatrix *matrix, FILE *out);

//SwapMap getRowSwapMap(COOMatrix *matrix);
//void swapRow(SwapMap *rowSwapMap);
//void swapRowInverse(SwapMap *rowSwapMap);

#endif //SPARSEMATRIX_COOMATRIX_H

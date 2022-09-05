#ifndef SPARSEMATRIX_COOMATRIX_H
#define SPARSEMATRIX_COOMATRIX_H


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mmio.h"
#include "Histogram.h"

typedef struct COOMatrix {
    float *data;
    u_int64_t *col_index;
    u_int64_t *row_index;
    u_int64_t row_size;
    u_int64_t col_size;
    u_int64_t num_non_zero_elements;
} COOMatrix;

COOMatrix *COOMatrix_new();
void COOMatrix_free(COOMatrix *matrix);
void COOMatrix_outAsJSON(const COOMatrix *matrix, FILE *out);
void COOMatrix_infoOutAsJSON(const COOMatrix *matrix, FILE *out);
u_int64_t COOMatrix_maxNumberOfElem(const COOMatrix *matrix);
int COOMatrix_split(const COOMatrix *matrix, COOMatrix *first, COOMatrix *second, u_int64_t threshold);

void COOMatrix_free_wpm(COOMatrix *matrix);
int COOMatrix_split_wpm(const COOMatrix *matrix, COOMatrix *first, COOMatrix *second, u_int64_t threshold);

#endif //SPARSEMATRIX_COOMATRIX_H

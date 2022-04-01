//
// Created by 9669c on 15/03/2022.
//

#ifndef SPARSEMATRIX_ELLMATRIX_H
#define SPARSEMATRIX_ELLMATRIX_H

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include <stdio.h>
#include <memory.h>

typedef struct ELLMatrix {
    float *data;
    int *col_index;
    size_t data_size;
    int num_elem;
    int row_size;
    int col_size;
    int num_non_zero_elements;
} ELLMatrix;

ELLMatrix *ELLMatrix_new(CSRMatrix *csrMatrix);
void ELLMatrix_free(ELLMatrix *ellMatrix);
void ELLMatrix_outAsJSON(ELLMatrix *matrix, FILE *out);



#endif //SPARSEMATRIX_ELLMATRIX_H

//
// Created by 9669c on 14/03/2022.
//

#ifndef SPARSEMATRIX_CSRMATRIX_H
#define SPARSEMATRIX_CSRMATRIX_H

#include <stdio.h>
#include <iostream>
#include <cstring>

#include "COOMatrix.h"


extern "C" {
#include "mmio.h"
};

class CSRMatrix {
private:
    float *data;
    int *col_index;
    int *row_pointer;
    int num_non_zero_elements;
    int row_size;
    int col_size;
public:
    CSRMatrix(COOMatrix &matrix);
    ~CSRMatrix();
    float *getData();
    int *getColIndex();
    int *getRowPointer();
    int getNumNonZeroElements();
    int getRowSize();
    int getColSize();
    void SpMV(float *X, float *Y);
    void SpMV_GPU(float *X, float *Y);
    void print();
};


#endif //SPARSEMATRIX_CSRMATRIX_H

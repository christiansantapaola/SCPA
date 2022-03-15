//
// Created by 9669c on 14/03/2022.
//

#ifndef SPARSEMATRIX_COOMATRIX_H
#define SPARSEMATRIX_COOMATRIX_H

#include <stdio.h>
#include <iostream>
#include <math.h>

extern "C" {
#include "mmio.h"
};

class COOMatrix {
private:
    float *data;
    int *col_index;
    int *row_index;
    int row_size;
    int col_size;
    int num_non_zero_elements;
public:
    COOMatrix(float *Matrix, int rows, int cols);
    COOMatrix(FILE *f);
    ~COOMatrix();
    float *getData();
    int *getColIndex();
    int *getRowIndex();
    int getNumNonZeroElements();
    int getRowSize();
    int getColSize();
    void print();
};

#endif //SPARSEMATRIX_COOMATRIX_H

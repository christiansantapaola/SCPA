//
// Created by 9669c on 14/03/2022.
//

#ifndef SPARSEMATRIX_CSRMATRIX_H
#define SPARSEMATRIX_CSRMATRIX_H

#include <iostream>
#include <cstring>
#include <chrono>

#include "COOMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"

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
    CSRMatrix() = default;
    CSRMatrix(COOMatrix &matrix);
    ~CSRMatrix();
    float *getData() const;
    int *getColIndex() const;
    int *getRowPointer() const;
    int getNumNonZeroElements() const;
    int getRowSize() const;
    int getColSize() const;
    friend std::ostream& operator<<(std::ostream &out, CSRMatrix const& matrix);
    SpMVResult SpMV_CPU(Vector &X, Vector &Y);
    SpMVResult SpMV_GPU(Vector &X, Vector &Y);
};


#endif //SPARSEMATRIX_CSRMATRIX_H

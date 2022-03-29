//
// Created by 9669c on 15/03/2022.
//

#ifndef SPARSEMATRIX_ELLMATRIX_H
#define SPARSEMATRIX_ELLMATRIX_H

#include <chrono>

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "SpMVResult.h"
#include "Vector.h"


class ELLMatrix {
private:
    float *data;
    int *col_index;
    size_t data_size;
    int num_elem;
    int row_size;
    int col_size;
    int num_non_zero_elements;
public:
    ELLMatrix() = default;
    ELLMatrix(CSRMatrix &matrix);
    ~ELLMatrix();
    float *getData();
    int *getColIndex();
    int getRowSize();
    int getColSize();
    int getNumNonZeroElements();
    SpMVResult SpMV_CPU(Vector &X, Vector &Y);
    SpMVResult SpMV_GPU(Vector &X, Vector &Y);
    friend std::ostream& operator<< (std::ostream &out, ELLMatrix const& matrix);

};


#endif //SPARSEMATRIX_ELLMATRIX_H

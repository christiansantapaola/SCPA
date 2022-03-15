//
// Created by 9669c on 15/03/2022.
//

#ifndef SPARSEMATRIX_ELLMATRIX_H
#define SPARSEMATRIX_ELLMATRIX_H

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "Vector.h"

class ELLMatrix {
private:
    float *data;
    int *col_index;
    int data_size;
    int ell_row;
    int ell_col;
    int row_size;
    int col_size;
    int num_non_zero_elements;
public:
    ELLMatrix(CSRMatrix &matrix);
    ~ELLMatrix();
    float *getData();
    int *getColIndex();
    int getEllRowSize();
    int getEllColSize();
    int getRowSize();
    int getColSize();
    int getNumNonZeroElements();
    void SpMV(Vector &X, Vector &Y);
    friend std::ostream& operator<< (std::ostream &out, ELLMatrix const& matrix);

};


#endif //SPARSEMATRIX_ELLMATRIX_H

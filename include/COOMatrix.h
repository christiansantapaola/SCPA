//
// Created by 9669c on 14/03/2022.
//

#ifndef SPARSEMATRIX_COOMATRIX_H
#define SPARSEMATRIX_COOMATRIX_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>


extern "C" {
#include "mmio.h"
};

#include "SwapMap.h"
#include "Histogram.h"

class COOMatrix {
private:
    float *data;
    int *col_index;
    int *row_index;
    int row_size;
    int col_size;
    int num_non_zero_elements;
public:
    COOMatrix() = default;
    COOMatrix(const float *Matrix, int rows, int cols);
    COOMatrix(FILE *f);
    ~COOMatrix();
    float *getData();
    int *getColIndex();
    int *getRowIndex();
    int getNumNonZeroElements() const;
    int getRowSize() const;
    int getColSize() const;
    SwapMap getRowSwapMap();
    void swapRow(SwapMap& rowSwapMap);
    void swapRowInverse(SwapMap& rowSwapMap);
    friend std::ostream& operator<< (std::ostream &out, COOMatrix const& matrix);

};

#endif //SPARSEMATRIX_COOMATRIX_H

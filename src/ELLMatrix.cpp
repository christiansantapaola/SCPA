//
// Created by 9669c on 15/03/2022.
//

#include "ELLMatrix.h"

ELLMatrix::ELLMatrix(CSRMatrix &matrix) {
    row_size = matrix.getRowSize();
    col_size = matrix.getColSize();
    num_non_zero_elements = matrix.getNumNonZeroElements();

    // find the the maximum number of non zero elements in a row.
    int max_num_nz_elem = 0;
    for (int row = 0; row < row_size; row++) {
        int num_nz_elem_curr_row = matrix.getRowPointer()[row + 1] - matrix.getRowPointer()[row];
        if (max_num_nz_elem < num_nz_elem_curr_row) {
            max_num_nz_elem = num_nz_elem_curr_row;
        }
    }

    ell_row = max_num_nz_elem;
    ell_col = row_size;
    data_size = ell_row * ell_col;

    data = new float[data_size];
    col_index = new int[data_size];
    // add padding;
    memset(data, 0, data_size * sizeof(float));
    memset(col_index, 0, data_size * sizeof(int));

    for (int row = 0; row < row_size; row++) {
        int row_start = matrix.getRowPointer()[row];
        int num_nz_elem = matrix.getRowPointer()[row + 1] - row_start;
        for (int col = 0; col < col_size; col++) {
            data[row * col_size + col] = (col < num_nz_elem) ? matrix.getData()[row_start + col] : 0.0f;
            col_index[row * col_size + col] = (col < num_nz_elem) ? matrix.getColIndex()[row_start + col] : 0;
        }
    }
}

ELLMatrix::~ELLMatrix() {
    delete data;
    delete col_index;
}

float *ELLMatrix::getData() {
    return data;
}

int *ELLMatrix::getColIndex() {
    return col_index;
}

int ELLMatrix::getRowSize() {
    return row_size;
}

int ELLMatrix::getColSize() {
    return col_size;
}

int ELLMatrix::getEllRowSize() {
    return ell_row;
}

int ELLMatrix::getEllColSize() {
    return ell_col;
}

int ELLMatrix::getNumNonZeroElements() {
    return num_non_zero_elements;
}

std::ostream &operator<<(std::ostream &out, ELLMatrix const &matrix) {
    out << matrix.row_size << " " << matrix.col_size << " " << matrix.num_non_zero_elements << std::endl;
    for (int row = 0; row < matrix.ell_row; row++) {
        for (int i = 0; i < matrix.ell_col; i++) {
            int index = row + i * matrix.ell_col;
            if (matrix.col_index[index] >= 0)
                out << row + 1 << " " << matrix.col_index[index] + 1 << " " << matrix.data[index] << std::endl;
        }
    }
    return out;
}

void ELLMatrix::SpMV(Vector &X, Vector &Y) {
    for (int row = 0; row < row_size; row++) {
        float dot = 0.0f;
        for (int col = 0; col < ell_col; col++) {
            int index = row *ell_col + col;
            dot += data[index] * X.getData()[col_index[index]];
        }
        Y.getData()[row] += dot;
    }
}
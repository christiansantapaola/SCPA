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

    ell_row = row_size;
    ell_col = max_num_nz_elem;
    data_size = ell_row * ell_col;

    data = new float[data_size];
    col_index = new int[data_size];
    // add padding;
    memset(data, 0, data_size * sizeof(float));
    memset(col_index, 0, data_size * sizeof(int));

    for (int row = 0; row < ell_row; row++) {
        int row_start = matrix.getRowPointer()[row];
        int num_nz_elem = matrix.getRowPointer()[row + 1] - row_start;
        int padding = max_num_nz_elem - num_nz_elem;
        int ell_row_start = row * max_num_nz_elem;
        for (int i = 0; i < ell_col; i++) {
            data[ell_row_start + i] = matrix.getData()[row_start + i];
            col_index[ell_row_start + i] = matrix.getColIndex()[row_start + i];
        }
        for (int i = 0; i < padding; i++) {
            data[ell_row_start + num_nz_elem + i] = 0.0f;
            col_index[ell_row_start + num_nz_elem + i] = 0;
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
    out << "{ " << std::endl;
    out << "row_size = " << matrix.row_size << ", " << std::endl;
    out << "col_size = " << matrix.col_size << ", " << std::endl;
    out << "ell_row = " << matrix.ell_row << ", " << std::endl;
    out << "ell_col = " << matrix.ell_col << ", " << std::endl;
    out << "data_size = " << matrix.data_size << ", " << std::endl;
    out << "data = [ ";
    for (int i = 0; i < matrix.data_size - 1; i++) {
        out << matrix.data[i] << ", ";
    }
    out << matrix.data[matrix.data_size - 1] << " ], " << std::endl;
    out << "col_index = [ ";
    for (int i = 0; i < matrix.data_size - 1; i++) {
        out << matrix.col_index[i] << ", ";
    }
    out << matrix.col_index[matrix.data_size - 1] << "], " << std::endl;
    out << "}" <<std::endl;
    return out;
}

void ELLMatrix::SpMV(Vector &X, Vector &Y) {
    if (X.getSize() != Y.getSize()) return;
    if (X.getSize() != row_size) return;
    for (int row = 0; row < ell_row; row++) {
        float dot = 0.0f;
        for (int i = 0; i < ell_col; i++) {
            int index = row * ell_col + i;
            dot += data[index] * X.getData()[col_index[index]];
        }
        Y.getData()[row] += dot;
    }
}
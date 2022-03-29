//
// Created by 9669c on 15/03/2022.
//

#include "ELLMatrix.h"

template<class T>
void transpose(T *dest, const T *src, int num_row, int num_col) {
    for (int row = 0; row < num_row; row++) {
        for (int col = 0; col < num_col; col++) {
            int srcIdx = row * num_col + col;
            int dstIdx = col * num_row + row;
            dest[dstIdx] = src[srcIdx];
        }
    }
}

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

    num_elem = max_num_nz_elem;
    data_size = row_size * num_elem;

    data = new float[data_size];
    col_index = new int[data_size];
    // add padding;
    memset(data, 0, data_size * sizeof(float));
    memset(col_index, 0, data_size * sizeof(int));
    float *temp_data = new float[data_size];
    int *temp_col_index = new int[data_size];
    for (int row = 0; row < row_size; row++) {
        int row_start = matrix.getRowPointer()[row];
        int num_nz_elem = matrix.getRowPointer()[row + 1] - row_start;
        int padding = max_num_nz_elem - num_nz_elem;
        for (int i = 0; i < num_nz_elem; i++) {
            int index = row * num_elem + i;
            temp_data[index] = matrix.getData()[row_start + i];
            temp_col_index[index] = matrix.getColIndex()[row_start + i];
        }
    }

    transpose<float>(data, temp_data, row_size, num_elem);
    transpose<int>(col_index, temp_col_index, row_size, num_elem);
    delete[] temp_data;
    delete[] temp_col_index;

}

ELLMatrix::~ELLMatrix() {
    delete[] data;
    delete[] col_index;
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


int ELLMatrix::getNumNonZeroElements() {
    return num_non_zero_elements;
}

std::ostream &operator<<(std::ostream &out, ELLMatrix const &matrix) {
    out << "{ " << std::endl;
    out << "\"row_size\": " << matrix.row_size << ", " << std::endl;
    out << "\"col_size\": " << matrix.col_size << ", " << std::endl;
    out << "\"num_elem\": " << matrix.num_elem << ", " << std::endl;
    out << "\"data_size\": " << matrix.data_size << ", " << std::endl;
    out << "\"num_non_zero_elements\": " << matrix.num_non_zero_elements << ", " << std::endl;
    out << "\"data\": [ ";
    for (int i = 0; i < matrix.data_size - 1; i++) {
        out << matrix.data[i] << ", ";
    }
    out << matrix.data[matrix.data_size - 1] << " ], " << std::endl;
    out << "\"col_index\": [ ";
    for (int i = 0; i < matrix.data_size - 1; i++) {
        out << matrix.col_index[i] << ", ";
    }
    out << matrix.col_index[matrix.data_size - 1] << "] " << std::endl;
    out << "}";
    return out;
}

SpMVResult ELLMatrix::SpMV_CPU(Vector &X, Vector &Y) {
    SpMVResult result = {false, 0, 0, 0, 0};
    if (X.getSize() != col_size && Y.getSize() != row_size) {
        result.success = false;
        return result;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < row_size; row++) {
        float dot = 0.0f;
        for (int i = 0; i < num_elem; i++) {
            int index = row + i * row_size;
            dot += data[index] * X.getData()[col_index[index]];
        }
        Y.getData()[row] += dot;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    result.success = true;
    std::chrono::duration<float> cputime = t1 - t0;
    result.CPUFunctionExecutionTime = cputime.count() * 1000;
    return result;
}
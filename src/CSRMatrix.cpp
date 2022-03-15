//
// Created by 9669c on 14/03/2022.
//

#include "CSRMatrix.h"

CSRMatrix::CSRMatrix(COOMatrix &matrix) {
    row_size = matrix.getRowSize();
    col_size = matrix.getColSize();
    num_non_zero_elements = matrix.getNumNonZeroElements();
    data = new float[num_non_zero_elements];
    col_index = new int[num_non_zero_elements];
    row_pointer = new int[row_size + 1];
    memset(row_pointer, 0, (row_size + 1) * sizeof(int));
    char *isInit = new char[row_size + 1];
    memset(isInit, 0, sizeof(*isInit) * (row_size + 1));
    for (int i = 0; i < num_non_zero_elements; i++) {
        float val = matrix.getData()[i];
        int col = matrix.getColIndex()[i];
        int row = matrix.getRowIndex()[i];
        data[i] = val;
        col_index[i] = col;
        if (isInit[row] == 0) {
            row_pointer[row] = i;
            isInit[row] = 1;
        }
    }
    // se una riga è vuota, il suo indice deve essere quello della riga successiva.
    // Precedentemente è stato assegnato un valore solo alle righe non vuote, qui assegname le altre.
    // Iteriamo al contrario, cosi possiamo far propagare il valore di riga in caso di multiple righe vuote.
    for (int row = row_size - 1; row > 0; row--) {
        if (!isInit[row]) {
            row_pointer[row] = row_pointer[row + 1];
        }
    }
    delete isInit;
    row_pointer[row_size] = num_non_zero_elements;
}

CSRMatrix::~CSRMatrix() {
    delete data;
    delete row_pointer;
    delete col_index;
}

float *CSRMatrix::getData() {
    return data;
}

int *CSRMatrix::getColIndex() {
    return col_index;
}

int *CSRMatrix::getRowPointer() {
    return row_pointer;
}

int CSRMatrix::getNumNonZeroElements() {
    return num_non_zero_elements;
}

int CSRMatrix::getRowSize() {
    return row_size;
}

int CSRMatrix::getColSize() {
    return col_size;
}

void CSRMatrix::print() {
    std::cout << row_size << " " << col_size << " " << num_non_zero_elements << std::endl;
    for (int row = 0; row < row_size; row++) {
        for (int i = row_pointer[row]; i < row_pointer[row + 1]; i++) {
            std::cout << row + 1 << " " << col_index[i] + 1 << data[i] << std::endl;
        }
    }
}

void CSRMatrix::SpMV(float *X, float *Y) {
    if (!X || !Y) return;
    for (int row = 0; row < row_size; row++) {
        float dot = 0.0f;
        int row_start = row_pointer[row];
        int row_end = row_pointer[row + 1];
        for (int elem = row_start; elem < row_end; elem++) {
            dot += data[elem] * X[col_index[elem]];
        }
        Y[row] += dot;
    }
}



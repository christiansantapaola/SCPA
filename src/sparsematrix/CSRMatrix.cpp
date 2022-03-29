//
// Created by 9669c on 14/03/2022.
//

#include "CSRMatrix.h"

//CSRMatrix::CSRMatrix(COOMatrix &matrix) {
//    row_size = matrix.getRowSize();
//    col_size = matrix.getColSize();
//    num_non_zero_elements = matrix.getNumNonZeroElements();
//    data = new float[num_non_zero_elements];
//    col_index = new int[num_non_zero_elements];
//    row_pointer = new int[row_size + 1];
//    memset(row_pointer, 0, (row_size + 1) * sizeof(int));
//    char *isInit = new char[row_size + 1];
//    memset(isInit, 0, sizeof(char) * (row_size + 1));
//    int lastpos = 0;
//    for (int row = 0; row < row_size; row++) {
//        for (int i = 0; i < num_non_zero_elements; i++) {
//            float val = matrix.getData()[i];
//            int col = matrix.getColIndex()[i];
//            int elem_row = matrix.getRowIndex()[i];
//            if (elem_row == row) {
//                data[lastpos] = val;
//                col_index[lastpos] = col;
//                if (isInit[elem_row] == 0) {
//                    row_pointer[elem_row] = lastpos;
//                    isInit[elem_row] = 1;
//                }
//                lastpos++;
//            }
//        }
//    }
//    // se una riga è vuota, il suo indice deve essere quello della riga successiva.
//    // Precedentemente è stato assegnato un valore solo alle righe non vuote, qui assegname le altre.
//    // Iteriamo al contrario, cosi possiamo far propagare il valore di riga in caso di multiple righe vuote.
//    for (int row = row_size - 1; row > 0; row--) {
//        if (!isInit[row]) {
//            row_pointer[row] = row_pointer[row + 1];
//        }
//    }
//    delete[] isInit;
//    row_pointer[row_size] = num_non_zero_elements;
//}


CSRMatrix::CSRMatrix(COOMatrix &matrix) {
    row_size = matrix.getRowSize();
    col_size = matrix.getColSize();
    num_non_zero_elements = matrix.getNumNonZeroElements();
    data = new float[num_non_zero_elements];
    col_index = new int[num_non_zero_elements];
    row_pointer = new int[row_size + 1];
    Histogram elemForRow = Histogram(row_size);

    // mi calcolo prima la posizione in base alle righe, poi aggiungo il resto,
    // questo perchè gli elementi in COO non devono essere ordinati.
    for (int i = 0; i < num_non_zero_elements; i++) {
        elemForRow.insert(matrix.getRowIndex()[i]);
    }
    int count = 0;
    for (int i = 0; i < row_size + 1; i++) {
        row_pointer[i] = count;
        count += elemForRow.getElemAtIndex(i);
    }


    /*
     * Qui uso un istogramma per salvarmi il numero di inserimenti alla riga i.
     */
    Histogram elemInsertedForRow = Histogram(row_size);
    for (int i = 0; i < num_non_zero_elements; i++) {
        int row = matrix.getRowIndex()[i];
        int col = matrix.getColIndex()[i];
        float val = matrix.getData()[i];
        int index = row_pointer[row] + elemInsertedForRow.getElemAtIndex(row);
        data[index] = val;
        col_index[index] = col;
        elemInsertedForRow.insert(row);
    }
}


CSRMatrix::~CSRMatrix() {
    delete[] data;
    delete[] row_pointer;
    delete[] col_index;
}

float *CSRMatrix::getData() const {
    return data;
}

int *CSRMatrix::getColIndex() const{
    return col_index;
}

int *CSRMatrix::getRowPointer() const {
    return row_pointer;
}

int CSRMatrix::getNumNonZeroElements() const {
    return num_non_zero_elements;
}

int CSRMatrix::getRowSize() const {
    return row_size;
}

int CSRMatrix::getColSize() const {
    return col_size;
}

std::ostream& operator<<(std::ostream &out, CSRMatrix const& matrix) {
    out << "{ " << std::endl;
    out << "\"row size\": " << matrix.row_size << "," << std::endl;
    out << "\"col size\": " << matrix.col_size << "," << std::endl;
    out << "\"num_non_zero_elements\":" << matrix.num_non_zero_elements << "," << std::endl;
    out << "\"row_pointer\": [ ";
    for (int i=0; i < matrix.row_size - 1; i++) {
        out << matrix.row_pointer[i] << ", ";
    }
    out << matrix.row_pointer[matrix.row_size - 1] << " ]" << std::endl;

    out << "\"col_index\": [ ";
    for (int i=0; i < matrix.num_non_zero_elements - 1; i++) {
        out << matrix.col_index[i] << ", ";
    }
    out << matrix.col_index[matrix.num_non_zero_elements - 1] << " ]" << std::endl;

    out << "\"data\": [ ";
    for (int i=0; i < matrix.num_non_zero_elements - 1; i++) {
        out << matrix.data[i] << ", ";
    }
    out << matrix.data[matrix.num_non_zero_elements - 1] << " ]" << std::endl;
    out << "}";
    return out;
}

SpMVResult CSRMatrix::SpMV_CPU(Vector &X, Vector &Y) {
    SpMVResult result = {false, 0, 0, 0, 0};
    if (X.getSize() != col_size && Y.getSize() != row_size) {
        return result;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < row_size; row++) {
        float dot = 0.0f;
        int row_start = row_pointer[row];
        int row_end = row_pointer[row + 1];
        for (int elem = row_start; elem < row_end; elem++) {
            dot += data[elem] * X.getData()[col_index[elem]];
        }
        Y.getData()[row] += dot;
    }
    auto t1 =  std::chrono::high_resolution_clock::now();
    result.success = true;
    result.CPUFunctionExecutionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return result;
}



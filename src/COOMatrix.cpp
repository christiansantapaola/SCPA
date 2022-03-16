//
// Created by 9669c on 14/03/2022.
//

#include "COOMatrix.h"

COOMatrix::COOMatrix(FILE *f) {
    int ret_code;
    MM_typecode matcode;
    int M, N, nz;
    int i, *I, *J;
    double *val;

    if (mm_read_banner(f, &matcode) != 0) {
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode) ) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
        exit(1);
    }


    /* reserve memory for matrices */
    row_size = M;
    col_size = N;
    num_non_zero_elements = nz;
    row_index = new int[num_non_zero_elements];
    col_index = new int[num_non_zero_elements];
    data = new float[num_non_zero_elements];


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %f\n", &col_index[i], &row_index[i], &data[i]);
        /* adjust from 1-based to 0-based */
        row_index[i]--;
        col_index[i]--;
    }
}

COOMatrix::COOMatrix(float *Matrix, int rows, int cols) {
    row_size = rows;
    col_size = cols;
    num_non_zero_elements = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value = Matrix[i * cols + j];
            if (fabsf(value) > 0.00001) {
               num_non_zero_elements++;
            }
        }
    }
    data = new float[num_non_zero_elements];
    col_index = new int[num_non_zero_elements];
    row_index = new int[num_non_zero_elements];
    int k = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value = Matrix[i * cols + j];
            if (fabsf(value) > 0.00001) {
                data[k] = value;
                row_index[k] = i;
                col_index[k] = j;
                k++;
            }
        }
    }
}

COOMatrix::~COOMatrix() {
    delete data;
    delete row_index;
    delete col_index;
}

float *COOMatrix::getData() {
    return data;
}

int *COOMatrix::getColIndex() {
    return col_index;
}

int *COOMatrix::getRowIndex() {
    return row_index;
}

int COOMatrix::getNumNonZeroElements() {
    return num_non_zero_elements;
}

int COOMatrix::getRowSize() {
    return row_size;
}

int COOMatrix::getColSize() {
    return col_size;
}

std::ostream& operator<< (std::ostream &out, COOMatrix const& matrix) {
    out << "{ " << std::endl;
    out << "row size = " << matrix.row_size << "," << std::endl;
    out << "col size = " << matrix.col_size << "," << std::endl;
    out << "num_non_zero_elements = " << matrix.num_non_zero_elements << "," << std::endl;
    out << "row_index = [ ";
    for (int i=0; i < matrix.num_non_zero_elements - 1; i++) {
        out << matrix.row_index[i] << ", ";
    }
    out << matrix.row_index[matrix.num_non_zero_elements - 1] << " ]" << std::endl;

    out << "col_index = [ ";
    for (int i=0; i < matrix.num_non_zero_elements - 1; i++) {
        out << matrix.col_index[i] << ", ";
    }
    out << matrix.col_index[matrix.num_non_zero_elements - 1] << " ]" << std::endl;

    out << "data = [ ";
    for (int i=0; i < matrix.num_non_zero_elements - 1; i++) {
        out << matrix.data[i] << ", ";
    }
    out << matrix.data[matrix.num_non_zero_elements - 1] << " ]," << std::endl;
    out << "}" << std::endl;
    return out;
}

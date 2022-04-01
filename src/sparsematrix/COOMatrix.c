#include "COOMatrix.h"


COOMatrix *COOMatrix_new(FILE *f) {
    int ret_code;
    MM_typecode matcode;
    int M, N, nz;
    int i, *I, *J;
    double *val;
    COOMatrix *matrix = NULL;

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
    matrix = (COOMatrix *) malloc(sizeof(COOMatrix));
    if (!matrix) {
        perror("COOMatrix_new(): ");
        return NULL;
    }
    matrix->row_size = M;
    matrix->col_size = N;
    matrix->num_non_zero_elements = nz;
    matrix->row_index = (int *) malloc(matrix->num_non_zero_elements * sizeof(int));
    if (!matrix->row_index) {
        perror("COOMatrix_new(): ");
        free(matrix);
        return NULL;
    }
    matrix->col_index = (int *) malloc(matrix->num_non_zero_elements * sizeof(int));
    if (!matrix->col_index) {
        perror("COOMatrix_new(): ");
        free(matrix->row_index);
        free(matrix);
        return NULL;
    }
    matrix->data =(float *) malloc(matrix->num_non_zero_elements * sizeof(float));
    if (!matrix->data) {
        free(matrix->col_index);
        free(matrix->row_index);
        free(matrix);
        perror("COOMatrix_new(): ");
        return NULL;
    }


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %f\n", &matrix->col_index[i], &matrix->row_index[i], &matrix->data[i]);
        /* adjust from 1-based to 0-based */
        matrix->row_index[i]--;
        matrix->col_index[i]--;
    }
    return matrix;
}

COOMatrix *newCOOMatrixFromMatrixArray(const float *Matrix, int rows, int cols) {
    /* reserve memory for matrices */
    COOMatrix *matrix = NULL;
    matrix = (COOMatrix *) malloc(sizeof(COOMatrix));
    if (!matrix) {
        perror("COOMatrix_new(): ");
        return NULL;
    }
    matrix->row_size = rows;
    matrix->col_size = cols;
    matrix->num_non_zero_elements = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value = Matrix[i * cols + j];
            if (fabsf(value) > 0.00001) {
                matrix->num_non_zero_elements++;
            }
        }
    }
    matrix->row_index = (int *) malloc(matrix->num_non_zero_elements * sizeof(int));
    if (!matrix->row_index) {
        perror("COOMatrix_new(): ");
        free(matrix);
        return NULL;
    }
    matrix->col_index = (int *) malloc(matrix->num_non_zero_elements * sizeof(int));
    if (!matrix->col_index) {
        perror("COOMatrix_new(): ");
        free(matrix->row_index);
        free(matrix);
        return NULL;
    }
    matrix->data =(float *) malloc(matrix->num_non_zero_elements * sizeof(float));
    if (!matrix->data) {
        free(matrix->col_index);
        free(matrix->row_index);
        free(matrix);
        perror("COOMatrix_new(): ");
        return NULL;
    }
    int k = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value = Matrix[i * cols + j];
            if (fabsf(value) > 0.00001) {
                matrix->data[k] = value;
                matrix->row_index[k] = i;
                matrix->col_index[k] = j;
                k++;
            }
        }
    }
    return matrix;
}

void COOMatrix_free(COOMatrix *matrix) {
    if (!matrix) return;
    free(matrix->data);
    free(matrix->col_index);
    free(matrix->row_index);
    free(matrix);
}

void COOMatrix_outAsJSON(COOMatrix *matrix, FILE *out) {
    if (!matrix || !out) return;
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %d,\n", "\"row size\"",  matrix->row_size);
    fprintf(out, "%s: %d,\n", "\"col size\"",  matrix->col_size);
    fprintf(out, "%s: %d,\n", "\"num_non_zero_elements\"",  matrix->num_non_zero_elements);
    fprintf(out, "%s: [ ", "\"row_index\"");
    for (int i=0; i < matrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%d, ", matrix->row_index[i]);
    }
    fprintf(out, "%d ],\n", matrix->row_index[matrix->num_non_zero_elements - 1]);
    fprintf(out, "%s: [ ", "\"col_index\"");
    for (int i=0; i < matrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%d, ", matrix->col_index[i]);
    }
    fprintf(out, "%d ],\n", matrix->col_index[matrix->num_non_zero_elements - 1]);
    fprintf(out, "%s: [ ", "\"data\"");
    for (int i=0; i < matrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%f, ", matrix->data[i]);
    }
    fprintf(out, "%f ]\n", matrix->data[matrix->num_non_zero_elements - 1]);
    fprintf(out, "%s", "}");
}

//SwapMap COOMatrix::getRowSwapMap() {
//    SwapMap rowSwapMap = SwapMap(row_size);
//    Histogram histogram = Histogram(row_size);
//    for (int i = 0; i < num_non_zero_elements; i++) {
//        histogram.insert(row_index[i]);
//    }
//    for (int i = 0; i < row_size; i++) {
//        rowSwapMap.setMapping(i, histogram.getElemAtIndex(i));
//    }
//    return rowSwapMap;
//}
//
//void COOMatrix::swapRow(SwapMap& rowSwapMap) {
//    for (int i = 0; i < num_non_zero_elements; i++) {
//        row_index[i] = rowSwapMap.getMapping(row_index[i]);
//    }
//}
//
//void COOMatrix::swapRowInverse(SwapMap& rowSwapMap) {
//    for (int i = 0; i < num_non_zero_elements; i++) {
//        row_index[i] = rowSwapMap.getInverse(row_index[i]);
//    }
//}
#include "COOMatrix.h"

COOMatrix *COOMatrix_new(FILE *f) {
    return NULL;
}

//COOMatrix *newCOOMatrixFromMatrixArray(const float *Matrix, int rows, int cols) {
//    /* reserve memory for matrices */
//    COOMatrix *matrix = NULL;
//    matrix = (COOMatrix *) malloc(sizeof(COOMatrix));
//    if (!matrix) {
//        perror("COOMatrix_new(): ");
//        return NULL;
//    }
//    matrix->row_size = rows;
//    matrix->col_size = cols;
//    matrix->num_non_zero_elements = 0;
//
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            float value = Matrix[i * cols + j];
//            if (fabsf(value) > 0.00001) {
//                matrix->num_non_zero_elements++;
//            }
//        }
//    }
//    matrix->row_index = (int *) malloc(matrix->num_non_zero_elements * sizeof(int));
//    if (!matrix->row_index) {
//        perror("COOMatrix_new(): ");
//        free(matrix);
//        return NULL;
//    }
//    matrix->col_index = (int *) malloc(matrix->num_non_zero_elements * sizeof(int));
//    if (!matrix->col_index) {
//        perror("COOMatrix_new(): ");
//        free(matrix->row_index);
//        free(matrix);
//        return NULL;
//    }
//    matrix->data =(float *) malloc(matrix->num_non_zero_elements * sizeof(float));
//    if (!matrix->data) {
//        free(matrix->col_index);
//        free(matrix->row_index);
//        free(matrix);
//        perror("COOMatrix_new(): ");
//        return NULL;
//    }
//    int k = 0;
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            float value = Matrix[i * cols + j];
//            if (fabsf(value) > 0.00001) {
//                matrix->data[k] = value;
//                matrix->row_index[k] = i;
//                matrix->col_index[k] = j;
//                k++;
//            }
//        }
//    }
//    return matrix;
//}

void COOMatrix_free(COOMatrix *matrix) {
    if (!matrix) return;
    free(matrix->data);
    free(matrix->col_index);
    free(matrix->row_index);
    free(matrix);
}

void COOMatrix_outAsJSON(COOMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
    }
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %lu,\n", "\"row size\"",  matrix->row_size);
    fprintf(out, "%s: %lu,\n", "\"col size\"",  matrix->col_size);
    fprintf(out, "%s: %lu,\n", "\"num_non_zero_elements\"",  matrix->num_non_zero_elements);
    fprintf(out, "%s: [ ", "\"row_index\"");
    for (u_int64_t i=0; i < matrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%lu, ", matrix->row_index[i]);
    }
    fprintf(out, "%lu ],\n", matrix->row_index[matrix->num_non_zero_elements - 1]);
    fprintf(out, "%s: [ ", "\"col_index\"");
    for (u_int64_t i=0; i < matrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%lu, ", matrix->col_index[i]);
    }
    fprintf(out, "%lu ],\n", matrix->col_index[matrix->num_non_zero_elements - 1]);
    fprintf(out, "%s: [ ", "\"data\"");
    for (u_int64_t i=0; i < matrix->num_non_zero_elements - 1; i++) {
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
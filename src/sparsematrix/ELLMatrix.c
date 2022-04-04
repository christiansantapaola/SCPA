//
// Created by 9669c on 15/03/2022.
//

#include "ELLMatrix.h"
#include <omp.h>

void transposef(float *restrict dest, const float *restrict src, int num_row, int num_col) {
#pragma omp parallel for schedule(dynamic, 256) shared(num_row, num_col, dest, src) default(none)
    for (int row = 0; row < num_row; row++) {
        for (int col = 0; col < num_col; col++) {
            int srcIdx = row * num_col + col;
            int dstIdx = col * num_row + row;
            dest[dstIdx] = src[srcIdx];
        }
    }
}

void transposei(int *restrict dest, const int *restrict src, int num_row, int num_col) {
#pragma omp parallel for schedule(dynamic, 256) shared(num_row, num_col, dest, src) default(none)
    for (int row = 0; row < num_row; row++) {
        for (int col = 0; col < num_col; col++) {
            int srcIdx = row * num_col + col;
            int dstIdx = col * num_row + row;
            dest[dstIdx] = src[srcIdx];
        }
    }
}


ELLMatrix *ELLMatrix_new(CSRMatrix *csrMatrix) {
    if (!csrMatrix) return NULL;
    ELLMatrix *ellMatrix = (ELLMatrix *) malloc(sizeof(ELLMatrix));
    ellMatrix->row_size = csrMatrix->row_size;
    ellMatrix->col_size = csrMatrix->col_size;
    ellMatrix->num_non_zero_elements = csrMatrix->num_non_zero_elements;

    // find the the maximum number of non zero elements in a row.
    int max_num_nz_elem = 0;
    for (int row = 0; row < csrMatrix->row_size; row++) {
        int num_nz_elem_curr_row = csrMatrix->row_pointer[row + 1] - csrMatrix->row_pointer[row];
        if (max_num_nz_elem < num_nz_elem_curr_row) {
            max_num_nz_elem = num_nz_elem_curr_row;
        }
    }

    ellMatrix->num_elem = max_num_nz_elem;
    ellMatrix->data_size = ellMatrix->row_size * ellMatrix->num_elem;

    ellMatrix->data = (float *) malloc(ellMatrix->data_size * sizeof(float));
    ellMatrix->col_index = (int *) malloc(ellMatrix->data_size * sizeof(int));
    // add padding;
    memset(ellMatrix->data, 0, ellMatrix->data_size * sizeof(float));
    memset(ellMatrix->col_index, 0, ellMatrix->data_size * sizeof(int));
    for (int row = 0; row < ellMatrix->row_size; row++) {
        int row_start = csrMatrix->row_pointer[row];
        int num_nz_elem = csrMatrix->row_pointer[row + 1] - row_start;
        for (int i = 0; i < num_nz_elem; i++) {
            int index = row * ellMatrix->num_elem + i;
            ellMatrix->data[index] = csrMatrix->data[row_start + i];
            ellMatrix->col_index[index] = csrMatrix->col_index[row_start + i];
        }
    }

    return ellMatrix;
}

void ELLMatrix_free(ELLMatrix *ellMatrix) {
    if (!ellMatrix) return;
    free(ellMatrix->data);
    free(ellMatrix->col_index);
    free(ellMatrix);
}

void ELLMatrix_transpose(ELLMatrix *ellMatrix) {
    float *temp_data = (float *) malloc(ellMatrix->data_size * sizeof(float));
    int *temp_col_index = (int *) malloc(ellMatrix->data_size * sizeof(int));
    memcpy(temp_data, ellMatrix->data, ellMatrix->data_size * sizeof(float));
    memcpy(temp_col_index, ellMatrix->col_index, ellMatrix->data_size * sizeof(int));
    transposef(ellMatrix->data, temp_data, ellMatrix->row_size, ellMatrix->num_elem);
    transposei(ellMatrix->col_index, temp_col_index, ellMatrix->row_size, ellMatrix->num_elem);
    free(temp_data);
    free(temp_col_index);
}

void ELLMatrix_outAsJSON(ELLMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
    }
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %d,\n", "\"row size\"", matrix->row_size);
    fprintf(out, "%s: %d,\n", "\"col size\"", matrix->col_size);
    fprintf(out, "%s: %d,\n", "\"num_non_zero_elements\"", matrix->num_non_zero_elements);
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


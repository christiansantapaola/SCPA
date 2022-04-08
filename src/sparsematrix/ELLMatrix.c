//
// Created by 9669c on 15/03/2022.
//

#include "ELLMatrix.h"
#include <omp.h>

void transposef(float *restrict dest, const float *restrict src, u_int64_t num_row, u_int64_t num_col) {
#pragma omp parallel for schedule(dynamic, 256) shared(num_row, num_col, dest, src) default(none)
    for (size_t row = 0; row < num_row; row++) {
        for (size_t col = 0; col < num_col; col++) {
            size_t srcIdx = row * num_col + col;
            size_t dstIdx = col * num_row + row;
            dest[dstIdx] = src[srcIdx];
        }
    }
}

void transpose_u_int64_t(u_int64_t *restrict dest, const u_int64_t *restrict src, u_int64_t num_row, u_int64_t num_col) {
#pragma omp parallel for schedule(dynamic, 256) shared(num_row, num_col, dest, src) default(none)
    for (size_t row = 0; row < num_row; row++) {
        for (size_t col = 0; col < num_col; col++) {
            size_t srcIdx = row * num_col + col;
            size_t dstIdx = col * num_row + row;
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
    unsigned int max_num_nz_elem = 0;
    for (unsigned int row = 0; row < csrMatrix->row_size; row++) {
        unsigned int num_nz_elem_curr_row = csrMatrix->row_pointer[row + 1] - csrMatrix->row_pointer[row];
        if (max_num_nz_elem < num_nz_elem_curr_row) {
            max_num_nz_elem = num_nz_elem_curr_row;
        }
    }

    ellMatrix->num_elem = max_num_nz_elem;
    ellMatrix->data_size = ellMatrix->row_size * ellMatrix->num_elem;

    ellMatrix->data = (float *) malloc(ellMatrix->data_size * sizeof(float));
    ellMatrix->col_index = (u_int64_t *) malloc(ellMatrix->data_size * sizeof(u_int64_t));
    // add padding;
    memset(ellMatrix->data, 0, ellMatrix->data_size * sizeof(float));
    memset(ellMatrix->col_index, 0, ellMatrix->data_size * sizeof(u_int64_t));
    for (unsigned int row = 0; row < ellMatrix->row_size; row++) {
        unsigned int row_start = csrMatrix->row_pointer[row];
        unsigned int num_nz_elem = csrMatrix->row_pointer[row + 1] - row_start;
        for (int i = 0; i < num_nz_elem; i++) {
            unsigned int index = row * ellMatrix->num_elem + i;
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
    u_int64_t *temp_col_index = (u_int64_t *) malloc(ellMatrix->data_size * sizeof(u_int64_t));
    memcpy(temp_data, ellMatrix->data, ellMatrix->data_size * sizeof(float));
    memcpy(temp_col_index, ellMatrix->col_index, ellMatrix->data_size * sizeof(u_int64_t));
    transposef(ellMatrix->data, temp_data, ellMatrix->row_size, ellMatrix->num_elem);
    transpose_u_int64_t(ellMatrix->col_index, temp_col_index, ellMatrix->row_size, ellMatrix->num_elem);
    free(temp_data);
    free(temp_col_index);
}

void ELLMatrix_outAsJSON(ELLMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
        return;
    }
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %lu,\n", "\"row size\"", matrix->row_size);
    fprintf(out, "%s: %lu,\n", "\"col size\"", matrix->col_size);
    fprintf(out, "%s: %lu,\n", "\"num_non_zero_elements\"", matrix->num_non_zero_elements);
    fprintf(out, "%s: [ ", "\"col_index\"");
    for (int i=0; i < matrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%lu, ", matrix->col_index[i]);
    }
    fprintf(out, "%lu ],\n", matrix->col_index[matrix->num_non_zero_elements - 1]);
    fprintf(out, "%s: [ ", "\"data\"");
    for (int i=0; i < matrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%f, ", matrix->data[i]);
    }
    fprintf(out, "%f ]\n", matrix->data[matrix->num_non_zero_elements - 1]);
    fprintf(out, "%s", "}");
}

void ELLMatrix_infoOutAsJSON(ELLMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
        return;
    }
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %lu,\n", "\"row size\"",  matrix->row_size);
    fprintf(out, "%s: %lu,\n", "\"col size\"",  matrix->col_size);
    fprintf(out, "%s: %lu,\n", "\"num_non_zero_elements\"",  matrix->num_non_zero_elements);
    fprintf(out, "%s: %lf\n", "\"density\"",  ( ((double) matrix->num_non_zero_elements) / ((double) (matrix->row_size * matrix->col_size))));
    fprintf(out, "%s", "}");
}


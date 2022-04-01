//
// Created by 9669c on 15/03/2022.
//

#include "ELLMatrix.h"

void transposef(float *dest, const float *src, int num_row, int num_col) {
    for (int row = 0; row < num_row; row++) {
        for (int col = 0; col < num_col; col++) {
            int srcIdx = row * num_col + col;
            int dstIdx = col * num_row + row;
            dest[dstIdx] = src[srcIdx];
        }
    }
}

void transposei(int *dest, const int *src, int num_row, int num_col) {
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
    ELLMatrix *ellMatrix = malloc(sizeof(ELLMatrix));
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
    float *temp_data = (float *) malloc(ellMatrix->data_size * sizeof(float));
    int *temp_col_index = (int *) malloc(ellMatrix->data_size * sizeof(int));
    for (int row = 0; row < ellMatrix->row_size; row++) {
        int row_start = csrMatrix->row_pointer[row];
        int num_nz_elem = csrMatrix->row_pointer[row + 1] - row_start;
        for (int i = 0; i < num_nz_elem; i++) {
            int index = row * ellMatrix->num_elem + i;
            temp_data[index] = csrMatrix->data[row_start + i];
            temp_col_index[index] = csrMatrix->col_index[row_start + i];
        }
    }

    transposef(ellMatrix->data, temp_data, ellMatrix->row_size, ellMatrix->num_elem);
    transposei(ellMatrix->col_index, temp_col_index, ellMatrix->row_size, ellMatrix->num_elem);
    free(temp_data);
    free(temp_col_index);
    return ellMatrix;
}

void ELLMatrix_free(ELLMatrix *ellMatrix) {
    if (!ellMatrix) return;
    free(ellMatrix->data);
    free(ellMatrix->col_index);
    free(ellMatrix);
}


void ELLMatrix_outAsJSON(ELLMatrix *ellMatrix, FILE *out) {
    if (!ellMatrix || !out) return;
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %d,\n", "\"row size\"",  ellMatrix->row_size);
    fprintf(out, "%s: %d,\n", "\"col size\"",  ellMatrix->col_size);
    fprintf(out, "%s: %d,\n", "\"num_non_zero_elements\"",  ellMatrix->num_non_zero_elements);
    fprintf(out, "%s: [ ", "\"col_index\"");
    for (int i=0; i < ellMatrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%d, ", ellMatrix->col_index[i]);
    }
    fprintf(out, "%d ],\n", ellMatrix->col_index[ellMatrix->num_non_zero_elements - 1]);
    fprintf(out, "%s: [ ", "\"data\"");
    for (int i=0; i < ellMatrix->num_non_zero_elements - 1; i++) {
        fprintf(out, "%f, ", ellMatrix->data[i]);
    }
    fprintf(out, "%f ]\n", ellMatrix->data[ellMatrix->num_non_zero_elements - 1]);
    fprintf(out, "%s", "}");
}


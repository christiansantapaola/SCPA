#include "ELLMatrix.h"

ELLMatrix *ELLMatrix_new(const CSRMatrix *csrMatrix) {
    if (!csrMatrix) return NULL;
    ELLMatrix *ellMatrix = (ELLMatrix *) malloc(sizeof(ELLMatrix));
    ellMatrix->row_size = csrMatrix->row_size;
    ellMatrix->col_size = csrMatrix->col_size;
    ellMatrix->num_non_zero_elements = csrMatrix->num_non_zero_elements;

    // find the the maximum number of non zero elements in a row.
    u_int64_t max_num_nz_elem = 0;
    for (u_int64_t row = 0; row < csrMatrix->row_size; row++) {
        u_int64_t num_nz_elem_curr_row = csrMatrix->row_pointer[row + 1] - csrMatrix->row_pointer[row];
        if (max_num_nz_elem < num_nz_elem_curr_row) {
            max_num_nz_elem = num_nz_elem_curr_row;
        }
    }

    ellMatrix->num_elem = max_num_nz_elem;
    ellMatrix->data_row_size = ellMatrix->row_size;
    ellMatrix->data_col_size = ellMatrix->num_elem;
    ellMatrix->data_size = ellMatrix->row_size * ellMatrix->num_elem;

    ellMatrix->data = (float *) malloc(ellMatrix->data_size * sizeof(float));
    ellMatrix->col_index = (u_int64_t *) malloc(ellMatrix->data_size * sizeof(u_int64_t));
    // add padding;
    memset(ellMatrix->data, 0, ellMatrix->data_size * sizeof(float));
    memset(ellMatrix->col_index, 0, ellMatrix->data_size * sizeof(u_int64_t));
    for (u_int64_t row = 0; row < ellMatrix->row_size; row++) {
        u_int64_t row_start = csrMatrix->row_pointer[row];
        u_int64_t num_nz_elem = csrMatrix->row_pointer[row + 1] - row_start;
        for (u_int64_t i = 0; i < num_nz_elem; i++) {
            u_int64_t index = row * ellMatrix->num_elem + i;
            ellMatrix->data[index] = csrMatrix->data[row_start + i];
            ellMatrix->col_index[index] = csrMatrix->col_index[row_start + i];
        }
    }

    return ellMatrix;
}

ELLMatrix *ELLMatrix_new_fromCOO(const COOMatrix *cooMatrix) {
    if (!cooMatrix) return NULL;
    ELLMatrix *ellMatrix = (ELLMatrix *) malloc(sizeof(ELLMatrix));
    ellMatrix->row_size = cooMatrix->row_size;
    ellMatrix->col_size = cooMatrix->col_size;
    ellMatrix->num_non_zero_elements = cooMatrix->num_non_zero_elements;

    // find the the maximum number of non zero elements in a row.
    ellMatrix->num_elem = COOMatrix_maxNumberOfElem(cooMatrix);
    ellMatrix->data_row_size = ellMatrix->row_size;
    ellMatrix->data_col_size = ellMatrix->num_elem;
    ellMatrix->data_size = ellMatrix->row_size * ellMatrix->num_elem;

    ellMatrix->data = (float *) malloc(ellMatrix->data_size * sizeof(float));
    ellMatrix->col_index = (u_int64_t *) malloc(ellMatrix->data_size * sizeof(u_int64_t));
    // add padding;
    memset(ellMatrix->data, 0, ellMatrix->data_size * sizeof(float));
    memset(ellMatrix->col_index, 0, ellMatrix->data_size * sizeof(u_int64_t));
    Histogram *elemInserted = Histogram_new(cooMatrix->row_size + 1);
    for (u_int64_t i = 0; i < cooMatrix->num_non_zero_elements; i++) {
        u_int64_t row = cooMatrix->row_index[i];
        u_int64_t col = cooMatrix->col_index[i];
        float data = cooMatrix->data[i];
        u_int64_t base = row * ellMatrix->num_elem;
        u_int64_t offset = Histogram_getElemAtIndex(elemInserted, row);
        ellMatrix->data[base + offset] = data;
        ellMatrix->col_index[base + offset] = col;
        Histogram_insert(elemInserted, row);
    }
    Histogram_free(elemInserted);
    return ellMatrix;
}


void ELLMatrix_free(ELLMatrix *ellMatrix) {
    if (!ellMatrix) return;
    free(ellMatrix->data);
    free(ellMatrix->col_index);
    free(ellMatrix);
}

void ELLMatrix_transpose(const ELLMatrix *ellMatrix, ELLMatrix *transposed) {
    if (!transposed) {
        return;
    }
    transposed->col_size = ellMatrix->row_size;
    transposed->row_size = ellMatrix->col_size;
    transposed->data_col_size = ellMatrix->data_row_size;
    transposed->data_row_size = ellMatrix->data_col_size;

    transposed->num_elem = ellMatrix->num_elem;
    transposed->num_non_zero_elements = ellMatrix->num_non_zero_elements;
    transposed->data = (float *) malloc(ellMatrix->data_size * sizeof(float));
    transposed->col_index = (u_int64_t *) malloc(ellMatrix->data_size * sizeof(u_int64_t));
    //memcpy(transposed->data, ellMatrix->data, ellMatrix->data_size * sizeof(float));
    //memcpy(transposed->col_index ellMatrix->col_index, ellMatrix->data_size * sizeof(u_int64_t));
    transposef(ellMatrix->data, transposed->data, ellMatrix->data_row_size, ellMatrix->data_col_size);
    transpose_u_int64_t(ellMatrix->col_index, transposed->col_index, ellMatrix->data_row_size, ellMatrix->data_col_size);
}

void ELLMatrix_outAsJSON(ELLMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
        return;
    }
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %lu,\n", "\"row size\"", matrix->row_size);
    fprintf(out, "%s: %lu,\n", "\"col size\"", matrix->col_size);
    fprintf(out, "%s: %lu,\n", "\"num_non_zero_elements\"", matrix->num_non_zero_elements);
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

void ELLMatrix_infoOutAsJSON(ELLMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
        return;
    }
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %lu,\n", "\"row size\"",  matrix->row_size);
    fprintf(out, "%s: %lu,\n", "\"col size\"",  matrix->col_size);
    fprintf(out, "%s: %lu,\n", "\"num_non_zero_elements\"",  matrix->num_non_zero_elements);
    fprintf(out, "%s: %lf\n", "\"density\"",  ( ((double) matrix->num_non_zero_elements) / ((double) (matrix->row_size * matrix->col_size))));
    fprintf(out, "%s", "}");
}


#include "COOMatrix.h"

COOMatrix *COOMatrix_new() {
    COOMatrix *matrix = malloc(sizeof(*matrix));
    if (!matrix) {
        return NULL;
    }
    memset(matrix, 0, sizeof(*matrix));
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

void COOMatrix_infoAsJSON(COOMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
    }
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %lu,\n", "\"row size\"",  matrix->row_size);
    fprintf(out, "%s: %lu,\n", "\"col size\"",  matrix->col_size);
    fprintf(out, "%s: %lu,\n", "\"num_non_zero_elements\"",  matrix->num_non_zero_elements);
    fprintf(out, "%s: %f,\n", "\"density\"",  (double) matrix->num_non_zero_elements / (double) (matrix->row_size * matrix->col_size));
    fprintf(out, "%s", "}");
}

u_int64_t COOMatrix_maxNumberOfElem(COOMatrix *matrix) {
    if (!matrix) return 0;
    Histogram *hist = Histogram_new(matrix->row_size);
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
        Histogram_insert(hist, matrix->row_index[i]);
    }
    u_int64_t max = 0;
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
        u_int64_t elem = Histogram_getElemAtIndex(hist, i);
        if (max < elem) {
            max = elem;
        }
    }
    Histogram_free(hist);
    return max;
}

int COOMatrix_split(const COOMatrix *matrix, COOMatrix *first, COOMatrix *second, u_int64_t threshold) {
    if (!matrix || !first || !second) return -1;
    Histogram *rowsElem = Histogram_new(matrix->row_size + 1);
    first->row_size = matrix->row_size;
    first->col_size = matrix->col_size;
    first->num_non_zero_elements = 0;
    second->row_size = matrix->row_size;
    second->col_size = matrix->col_size;
    second->num_non_zero_elements = 0;
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
        Histogram_insert(rowsElem, matrix->row_index[i]);
    }
    for (u_int64_t i = 0; i < matrix->row_size; i++) {
        u_int64_t numElem = Histogram_getElemAtIndex(rowsElem, i);
        if (numElem <= threshold) {
            first->num_non_zero_elements += numElem;
        } else {
            second->num_non_zero_elements += numElem;
        }
    }
    if (first->num_non_zero_elements == 0 || second->num_non_zero_elements == 0) {
        Histogram_free(rowsElem);
        return 1;
    }
    first->row_index = malloc(first->num_non_zero_elements * sizeof(u_int64_t));
    first->col_index = malloc(first->num_non_zero_elements * sizeof(u_int64_t));
    first->data = malloc(first->num_non_zero_elements * sizeof(float));
    memset(first->data, 0,first->num_non_zero_elements * sizeof(float ) );
    memset(first->row_index, 0,first->num_non_zero_elements * sizeof(u_int64_t) );
    memset(first->col_index, 0,first->num_non_zero_elements * sizeof(u_int64_t) );



    second->row_index = malloc(second->num_non_zero_elements * sizeof(u_int64_t));
    second->col_index = malloc(second->num_non_zero_elements * sizeof(u_int64_t));
    second->data = malloc(second->num_non_zero_elements * sizeof(float));

    memset(second->data, 0,second->num_non_zero_elements * sizeof(float ) );
    memset(second->row_index, 0,second->num_non_zero_elements * sizeof(u_int64_t) );
    memset(second->col_index, 0,second->num_non_zero_elements * sizeof(u_int64_t) );


    u_int64_t fpos = 0, spos = 0;
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
        u_int64_t numElem = Histogram_getElemAtIndex(rowsElem, matrix->row_index[i]);
        if (numElem <= threshold) {
            first->row_index[fpos] = matrix->row_index[i];
            first->col_index[fpos] = matrix->col_index[i];
            first->data[fpos] = matrix->data[i];
            fpos++;
        } else {
            second->row_index[spos] = matrix->row_index[i];
            second->col_index[spos] = matrix->col_index[i];
            second->data[spos] = matrix->data[i];
            spos++;
        }
    }
    Histogram_free(rowsElem);
    return 0;
}

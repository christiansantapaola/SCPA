#include "COOMatrix.h"

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
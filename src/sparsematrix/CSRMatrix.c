//
// Created by 9669c on 14/03/2022.
//

#include "CSRMatrix.h"

CSRMatrix *CSRMatrix_new(COOMatrix *cooMatrix) {
    if (!cooMatrix) return NULL;
    CSRMatrix *csrMatrix = NULL;
    csrMatrix = (CSRMatrix *) malloc(sizeof(CSRMatrix));
    csrMatrix->row_size = cooMatrix->row_size;
    csrMatrix->col_size = cooMatrix->col_size;
    csrMatrix->num_non_zero_elements = cooMatrix->num_non_zero_elements;
    csrMatrix->data = malloc(csrMatrix->num_non_zero_elements * sizeof(float ));
    csrMatrix->col_index = malloc(csrMatrix->num_non_zero_elements * sizeof(u_int64_t));
    csrMatrix->row_pointer = malloc((csrMatrix->row_size + 1) * sizeof (u_int64_t));
    Histogram *elemForRow = Histogram_new(csrMatrix->row_size);

    // mi calcolo prima la posizione in base alle righe, poi aggiungo il resto,
    // questo perchè gli elementi in COO non devono essere ordinati.
    for (u_int64_t i = 0; i < csrMatrix->num_non_zero_elements; i++) {
        Histogram_insert(elemForRow, cooMatrix->row_index[i]);
    }
    u_int64_t count = 0;
    for (u_int64_t i = 0; i < cooMatrix->row_size + 1; i++) {
        csrMatrix->row_pointer[i] = count;
        count += Histogram_getElemAtIndex(elemForRow, i);
    }
    /*
     * Qui uso una bucket-list (struct Histogram) per salvarmi il numero di inserimenti alla riga i.
     */
    Histogram *elemInsertedForRow = Histogram_new(csrMatrix->row_size);
    for (u_int64_t i = 0; i < csrMatrix->num_non_zero_elements; i++) {
        u_int64_t row = cooMatrix->row_index[i];
        u_int64_t col = cooMatrix->col_index[i];
        float val = cooMatrix->data[i];
        u_int64_t offset = Histogram_getElemAtIndex(elemInsertedForRow, row);
        u_int64_t index = csrMatrix->row_pointer[row] + offset;
        csrMatrix->data[index] = val;
        csrMatrix->col_index[index] = col;
        Histogram_insert(elemInsertedForRow, row);
    }
    Histogram_free(elemForRow);
    Histogram_free(elemInsertedForRow);
    return csrMatrix;
}

void CSRMatrix_free(CSRMatrix *matrix) {
    if (!matrix) return;
    free(matrix->data);
    free(matrix->row_pointer);
    free(matrix->col_index);
    free(matrix);
}

void CSRMatrix_outAsJSON(CSRMatrix *matrix, FILE *out) {
    if (!out) out=stdout;
    if (!matrix) {
        fprintf(out, "{}");
        return;
    }
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %lu,\n", "\"row size\"",  matrix->row_size);
    fprintf(out, "%s: %lu,\n", "\"col size\"",  matrix->col_size);
    fprintf(out, "%s: %lu,\n", "\"num_non_zero_elements\"",  matrix->num_non_zero_elements);
    fprintf(out, "%s: [ ", "\"row_pointer\"");
    for (u_int64_t i=0; i < matrix->row_size; i++) {
        fprintf(out, "%lu, ", matrix->row_pointer[i]);
    }
    fprintf(out, "%lu ],\n", matrix->row_pointer[matrix->row_size]);
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

void CSRMatrix_infoOutAsJSON(CSRMatrix *matrix, FILE *out) {
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



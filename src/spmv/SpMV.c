#include "SpMV.h"

int COOMatrix_SpMV(const COOMatrix *matrix, const Vector *x, Vector *y, float *time) {
    clock_t start, end;
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }

    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        return SPMV_FAIL;
    }
    start = clock();
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
        y->data[matrix->row_index[i]] += matrix->data[i] * x->data[matrix->col_index[i]];
    }
    end = clock();
    if (time) {
        *time = ((float) (end - start)) / (float) CLOCKS_PER_SEC * 1000.0f;
    }
    return SPMV_SUCCESS;
}



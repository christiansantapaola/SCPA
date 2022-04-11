#include "SpMV.h"
#include <omp.h>

void CSRMatrix_SpMV_OPENMP(const CSRMatrix *matrix,const Vector *x, Vector *y, SpMVResultCPU *result) {
    double start, end;
    if (!matrix || !x || !y) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (result) {
        memset(result, 0, sizeof(*result));
    }
    start = omp_get_wtime();
#pragma omp parallel for schedule(auto) default(none) shared(matrix, x, y)
    for (u_int64_t row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        u_int64_t row_start = matrix->row_pointer[row];
        u_int64_t row_end = matrix->row_pointer[row + 1];
        for (u_int64_t elem = row_start; elem < row_end; elem++) {
            dot += matrix->data[elem] * x->data[matrix->col_index[elem]];
        }
        y->data[row] += dot;
    }
    end = omp_get_wtime();
    if (result) {
        result->success = 1;
        result->timeElapsed = (end - start) * 1000.0;
    }
}
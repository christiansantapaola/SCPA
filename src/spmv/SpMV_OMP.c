#include "SpMV.h"
#include <omp.h>

int CSRMatrix_SpMV(const CSRMatrix* matrix, const Vector* x, Vector* y, float* time, int parallel) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        return SPMV_FAIL;
    }
    double start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 256) shared(y, x, matrix) if (parallel)
    for (u_int64_t row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        int row_start = matrix->row_pointer[row];
        int row_end = matrix->row_pointer[row + 1];
        for (int elem = row_start; elem < row_end; elem++) {
            dot += matrix->data[elem] * x->data[matrix->col_index[elem]];
        }
        y->data[row] += dot;
    }
    double end = omp_get_wtime();
    if (time) {
        *time = (end - start) * 1000.0;
    }
    return SPMV_SUCCESS;
}

int ELLMatrix_SpMV(const ELLMatrix* matrix, const Vector* x, Vector* y, float* time, int parallel) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }

    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        return SPMV_FAIL;
    }

    double start = omp_get_wtime();

#pragma omp parallel for schedule(dynamic, 256) shared(y, x, matrix) if (parallel)
    for (u_int64_t row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        for (u_int64_t i = 0; i < matrix->num_elem; i++) {
            u_int64_t index = row * matrix->num_elem + i;
            dot += matrix->data[index] * x->data[matrix->col_index[index]];
        }
       y->data[row] += dot;
    }
    double end = omp_get_wtime();
    if (time) {
        *time = (end - start) * 1000.0;
    }

    return SPMV_SUCCESS;
}

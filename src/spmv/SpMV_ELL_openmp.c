#include "SpMV.h"
#include <omp.h>

void ELLMatrix_SpMV_OPENMP(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResultCPU *result) {
    double start, end;
    if (!matrix || !x || !y) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    if (result) {
        memset(result, 0, sizeof(*result));
        // memset(&result->blockGridInfo, 0, sizeof(result->blockGridInfo));
    }

    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        if (result) {
            result->success = 0;
        }
        return;
    }
    start = omp_get_wtime();
#pragma omp parallel for schedule(auto) default(none) shared(matrix, x, y, stderr)
    for (int row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        for (size_t i = 0; i < matrix->num_elem; i++) {
            size_t index = row * matrix->num_elem + i;
            dot += matrix->data[index] * x->data[matrix->col_index[index]];
        }
        y->data[row] += dot;
    }
    end = omp_get_wtime();

    if (result) {
        result->success = 1;
        result->timeElapsed = (end - start) * 1000.0;
    }
}
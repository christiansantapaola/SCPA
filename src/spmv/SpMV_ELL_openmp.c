#include "SpMV.h"
#include <omp.h>

void ELLMatrix_SpMV_OPENMP(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResultCPU *result) {
    clock_t start, end;
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
    start = clock();
#pragma omp parallel for schedule(dynamic) default(shared)
    for (int row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        for (int i = 0; i < matrix->num_elem; i++) {
            int index = row * matrix->num_elem + i;
            dot += matrix->data[index] * x->data[matrix->col_index[index]];
        }
        y->data[row] += dot;
    }
    end = clock();

    if (result) {
        result->success = 1;
        result->timeElapsed = ((float)(end - start)) / CLOCKS_PER_SEC * 1000.0f;
    }
}
#include "SpMV.h"
#include "COOMatrix.h"
#include <omp.h>

void COOMatrix_SpMV_OPENMP(const COOMatrix *matrix,const Vector *x, Vector *y, SpMVResultCPU *result) {
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
#pragma omp parallel for schedule(auto) default(shared)
    for (u_int64_t i = 0; i < matrix->num_non_zero_elements; i++) {
#pragma omp critical
        {
        y->data[matrix->row_index[i]] += matrix->data[i] * x->data[matrix->col_index[i]];
        }
    }
    end = clock();

    if (result) {
        result->success = 1;
        result->timeElapsed = ((float) (end - start)) / CLOCKS_PER_SEC * 1000.0f;
    }
}

#include "SpMV.h"

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
#pragma omp parallel for schedule(static) default(none) shared(matrix, x, y, stderr)
    for (int row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        for (size_t i = 0; i < matrix->num_elem; i++) {
            size_t index = row * matrix->num_elem + i;
            dot += matrix->data[index] * x->data[matrix->col_index[index]];
            //fprintf(stderr, "thread: %d/%d, row: %u, dot: %f, matrix->data[%u]: %f\n", omp_get_thread_num(), omp_get_num_threads(), row, dot, index, matrix->data[index]);
        }
        //fprintf(stderr, "thread: %d/%d, row: %u, dot: %f,\n", omp_get_thread_num(), omp_get_num_threads(), row, dot);
        y->data[row] += dot;
    }
    end = clock();

    if (result) {
        result->success = 1;
        result->timeElapsed = ((float)(end - start)) / CLOCKS_PER_SEC * 1000.0f;
    }
}
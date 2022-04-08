#include "SpMV.h"

void CSRMatrix_SpMV_CPU(const CSRMatrix *matrix,const Vector *x, Vector *y, SpMVResultCPU *result) {
    clock_t start, end;
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
    start = clock();
    for (int row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        int row_start = matrix->row_pointer[row];
        int row_end = matrix->row_pointer[row + 1];
        for (int elem = row_start; elem < row_end; elem++) {
            dot += matrix->data[elem] * x->data[matrix->col_index[elem]];
        }
       y->data[row] += dot;
    }
    end = clock();
    if (result) {
        result->success = 1;
        result->timeElapsed = ((float) (end - start)) / CLOCKS_PER_SEC * 1000.0f;
    }
}

void ELLMatrix_SpMV_CPU(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResultCPU *result) {
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
    for (int row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        for (int i = 0; i < matrix->num_elem; i++) {
            int index = row * matrix->num_elem + i;
            dot += matrix->data[index] * x->data[matrix->col_index[index]];
            //fprintf(stderr, "thread: %d/%d, row: %u, dot: %f, matrix->data[%u]: %f\n", 0,0, row, dot, index, matrix->data[index]);
        }
        //fprintf(stderr, "thread: %d/%d, row: %u, dot: %f,\n", 0, 0, row, dot);
        y->data[row] += dot;
    }
    end = clock();

    if (result) {
        result->success = 1;
        result->timeElapsed = ((float) (end - start)) / CLOCKS_PER_SEC * 1000.0f;
    }
}


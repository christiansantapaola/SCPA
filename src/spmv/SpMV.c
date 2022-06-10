#include "SpMV.h"

int CSRMatrix_SpMV(const CSRMatrix *matrix, const Vector *x, Vector *y, float *time, int parallel) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        return SPMV_FAIL;
    }
    #pragma omp parallel for schedule(auto) if (parallel)
    for (u_int64_t row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        int row_start = matrix->row_pointer[row];
        int row_end = matrix->row_pointer[row + 1];
        for (int elem = row_start; elem < row_end; elem++) {
            dot += matrix->data[elem] * x->data[matrix->col_index[elem]];
        }
       y->data[row] += dot;
    }
    return SPMV_SUCCESS;
}

int ELLMatrix_SpMV(const ELLMatrix *matrix, const Vector *x, Vector *y, float *time, int parallel) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }

    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        return SPMV_FAIL;
    }

    #pragma omp parallel for schedule(auto) if (parallel)
    for (u_int64_t row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        for (u_int64_t i = 0; i < matrix->num_elem; i++) {
            u_int64_t index = row * matrix->num_elem + i;
            dot += matrix->data[index] * x->data[matrix->col_index[index]];
        }
        y->data[row] += dot;
    }
    return SPMV_SUCCESS;
}


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


int ELLCOOMatrix_SpMV(COOMatrix *h_cooMatrix, Vector *h_x, Vector *h_y, u_int64_t threshold, u_int64_t max_iteration) {
        COOMatrix *h_low, *h_high;
        h_low = COOMatrix_new();
        h_high = COOMatrix_new();
        int notSplit = COOMatrix_split(h_cooMatrix, h_low, h_high, threshold);
        if (notSplit == -1) {
            return EXIT_FAILURE;
        }
        double totTime = 0.0;
        if (notSplit) {
            ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO(h_cooMatrix);
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
            int cudaDev = CudaUtils_getBestDevice(d_ellMatrix->data_size * sizeof(float) + (h_x->size + h_y->size) * sizeof(float));
            CudaUtils_setDevice(cudaDev);
            for (u_int64_t i = 0; i < max_iteration; i++) {
                float time;
                ELLMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, h_x, h_y, &time);
                totTime += time;
            }
            ELLMatrix_free_CUDA(d_ellMatrix);
            ELLMatrix_free(h_ellMatrix);
        } else {
            ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO(h_low);
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
            int cudaDev = CudaUtils_getBestDevice(d_ellMatrix->data_size * sizeof(float) + (h_x->size + h_y->size) * sizeof(float));
            CudaUtils_setDevice(cudaDev);
            for (u_int64_t i = 0; i < max_iteration; i++) {
                float time;
                ELLCOOMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, h_high, h_x, h_y, &time);
                totTime += time;
            }
            ELLMatrix_free_CUDA(d_ellMatrix);
            ELLMatrix_free(h_ellMatrix);
        }
        COOMatrix_free(h_low);
        COOMatrix_free(h_high);
        return SPMV_SUCCESS;

}
#include "SpMV.h"
#include <omp.h>


int CSRMatrix_SpMV(const CSRMatrix *matrix, const Vector *x, Vector *y, float *time, int parallel) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (x->size != matrix->col_size && y->size != matrix->row_size) {
        return SPMV_FAIL;
    }
    Vector **tmp = NULL;
    #pragma omp parallel if (parallel)
    {
        int num_threads = omp_get_num_threads();
        Vector *tmp_local = NULL;
        printf("num_treads = %d\n", num_threads);
        tmp = malloc(sizeof(*tmp) * num_threads);
        for (int i = 0; i < num_threads; i++) {
            tmp[i] = Vector_new(y->size);
            Vector_set(tmp[i], 0.0f);
        }
        printf("before for\n");
        
        printf("tmp_local = %p\n", tmp_local);
        printf("omp_get_thread_num = %d\n", num_threads);
        double start = omp_get_wtime();
        #pragma omp for schedule(dynamic, 256) private(tmp_local) 
        for (u_int64_t row = 0; row < matrix->row_size; row++) {
            tmp_local = tmp[omp_get_thread_num()];
            float dot = 0.0f;
            int row_start = matrix->row_pointer[row];
            int row_end = matrix->row_pointer[row + 1];
            for (int elem = row_start; elem < row_end; elem++) {
                dot += matrix->data[elem] * x->data[matrix->col_index[elem]];
            }
        tmp_local->data[row] += dot;
        }

        #pragma omp single 
        {
            printf("before sum\n");
            for (int i = 0; i < num_threads; i++) {
                Vector_sum(y, tmp[i]);
            }
            double end = omp_get_wtime();
            if (time) {
                *time = (end - start) * 1000.0;
            }
            printf("before free\n");
            for (int i = 0; i < num_threads; i++) {
                Vector_free(tmp[i]);
            }
            free(tmp);
            printf("after for\n");

        }
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

    int num_threads = omp_get_num_threads();
    Vector **tmp = malloc(sizeof(*tmp) * num_threads);
    if (!tmp) {
        return SPMV_FAIL;
    }

    for (int i = 0; i < num_threads; i++) {
        tmp[i] = Vector_new(y->size);
        Vector_set(tmp[i], 0.0f);
    }

    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(dynamic, 256) if (parallel)
    for (u_int64_t row = 0; row < matrix->row_size; row++) {
        float dot = 0.0f;
        for (u_int64_t i = 0; i < matrix->num_elem; i++) {
            u_int64_t index = row * matrix->num_elem + i;
            dot += matrix->data[index] * x->data[matrix->col_index[index]];
        }
        tmp[omp_get_thread_num()]->data[row] += dot;
    }
    for (int i = 0; i < num_threads; i++) {
        Vector_sum(y, tmp[i]);
    }

    double end = omp_get_wtime();
    if (time) {
        *time = (end - start) * 1000.0;
    }

    for (int i = 0; i < num_threads; i++) {
        Vector_free(tmp[i]);
    }
    free(tmp);

    return SPMV_SUCCESS;
}


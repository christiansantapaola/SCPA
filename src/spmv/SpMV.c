#include "SpMV.h"

int COOMatrix_SpMV(const COOMatrix* matrix, const Vector* x, Vector* y, Benchmark* benchmark) {
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
    if (benchmark) {
        benchmark->gpuTime = 0.0;
        benchmark->cpuTime = ((double)(end - start)) / (double)CLOCKS_PER_SEC;
    }
    return SPMV_SUCCESS;
}

int CSRMatrix_SpMV(const COOMatrix* matrix, const Vector* x, Vector* y, u_int64_t N, Benchmark* benchmark) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (benchmark) {
        benchmark->cpuTime = 0.0;
        benchmark->gpuTime = 0.0;
    }
    CSRMatrix* csrMatrix = CSRMatrix_new_wpm(matrix);
    int cudaDev = CudaUtils_getBestDevice(csrMatrix->num_non_zero_elements * sizeof(float) + (x->size + y->size) * sizeof(float));
    CudaUtils_setDevice(cudaDev);
    CSRMatrix* d_csrMatrix = CSRMatrix_to_CUDA(csrMatrix);
    for (u_int64_t i = 0; i < N; i++) {
        float time;
        CSRMatrix_SpMV_CUDA(cudaDev, d_csrMatrix, x, y, &time);
        if (benchmark) {
            benchmark->gpuTime += (double)time;
        }
    }
    CSRMatrix_free_CUDA(d_csrMatrix);
    CSRMatrix_free_wpm(csrMatrix);
    return SPMV_SUCCESS;
}

int ELLMatrix_SpMV(const COOMatrix* cooMatrix, const Vector* x, Vector* y, u_int64_t N, Benchmark* benchmark) {
    if (!cooMatrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (benchmark) {
        benchmark->cpuTime = 0.0;
        benchmark->gpuTime = 0.0;
    }
    COOMatrix* high, * low;
    high = COOMatrix_new();
    low = COOMatrix_new();
    int notSplit = COOMatrix_split(cooMatrix, low, high, 64);
    if (notSplit) {
        ELLMatrix* ellMatrix = ELLMatrix_new_fromCOO_wpm(cooMatrix);
        ELLMatrix* d_ellMatrix = ELLMatrix_to_CUDA(ellMatrix);
        int cudaDev = CudaUtils_getBestDevice(ellMatrix->data_size * sizeof(float) + (x->size + y->size) * sizeof(float));
        CudaUtils_setDevice(cudaDev);
        for (u_int64_t i = 0; i < N; i++) {
            float time;
            ELLMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, x, y, &time);
            if (benchmark) {
                benchmark->gpuTime += (double)time;
            }
        }
        ELLMatrix_free_CUDA(d_ellMatrix);
        ELLMatrix_free_wpm(ellMatrix);
    }
    else {
        ELLMatrix* ellMatrix = ELLMatrix_new_fromCOO_wpm(low);
        ELLMatrix* d_ellMatrix = ELLMatrix_to_CUDA(ellMatrix);
        int cudaDev = CudaUtils_getBestDevice(ellMatrix->data_size * sizeof(float) + (x->size + y->size) * sizeof(float));
        CudaUtils_setDevice(cudaDev);
        for (u_int64_t i = 0; i < N; i++) {
            float time;
            ELLCOOMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, high, x, y, &time);
            if (benchmark) {
                benchmark->gpuTime += (double)time;
            }
        }
        ELLMatrix_free_CUDA(d_ellMatrix);
        ELLMatrix_free_wpm(ellMatrix);
    }
    COOMatrix_free(high);
    COOMatrix_free(low);
    return SPMV_SUCCESS;
}

int CSRMatrix_SpMV_cpu(const COOMatrix* matrix, const Vector* x, Vector* y, u_int64_t N, Benchmark* benchmark) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (benchmark) {
        benchmark->cpuTime = 0.0;
        benchmark->gpuTime = 0.0;
    }
    CSRMatrix* csrMatrix = CSRMatrix_new(matrix);
    Benchmark tmp;
    for (u_int64_t i = 0; i < N; i++) {
        CSRMatrix_SpMV_OMP(csrMatrix, x, y, &tmp);
        if (benchmark) {
            benchmark->gpuTime += tmp.gpuTime;
            benchmark->cpuTime += tmp.cpuTime;
        }
    }
    CSRMatrix_free(csrMatrix);
    return SPMV_SUCCESS;
}

int ELLMatrix_SpMV_cpu(const COOMatrix* matrix, const Vector* x, Vector* y, u_int64_t N, Benchmark* benchmark) {
    if (!matrix || !x || !y) {
        return SPMV_FAIL;
    }
    if (benchmark) {
        benchmark->cpuTime = 0.0;
        benchmark->gpuTime = 0.0;
    }
    COOMatrix* first = COOMatrix_new();
    COOMatrix* second = COOMatrix_new();
    int notSplit = COOMatrix_split(matrix, first, second, 64);
    if (notSplit) {
        ELLMatrix* ellMatrix = ELLMatrix_new_fromCOO(matrix);
        Benchmark tmp;
        for (u_int64_t i = 0; i < N; i++) {
            ELLMatrix_SpMV_OMP(ellMatrix, x, y, &tmp);
            if (benchmark) {
                benchmark->gpuTime += tmp.gpuTime;
                benchmark->cpuTime += tmp.cpuTime;
            }
        }
        ELLMatrix_free(ellMatrix);
    }
    else {
        ELLMatrix* ellMatrix = ELLMatrix_new_fromCOO(first);
        Benchmark ellTmp, cooTmp;
        for (u_int64_t i = 0; i < N; i++) {
#pragma omp parallel sections
            {
#pragma omp section
                {
                    ELLMatrix_SpMV_OMP(ellMatrix, x, y, &ellTmp);

                }
#pragma omp section
                {
                    COOMatrix_SpMV(second, x, y, &cooTmp);
                }
            }
            if (benchmark) {
                benchmark->gpuTime += cooTmp.gpuTime + ellTmp.gpuTime;
                benchmark->cpuTime += cooTmp.cpuTime + ellTmp.cpuTime;
            }
        }
        ELLMatrix_free(ellMatrix);
    }
    COOMatrix_free(first);
    COOMatrix_free(second);
    return SPMV_SUCCESS;

}

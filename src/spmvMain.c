#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "SpMV.h"
#include "util.h"


int do_CSRMatrix_prod(char *path, struct dirent *entry, void *args) {
    (void) entry;
    (void) args;
    COOMatrix *cooMatrix = read_matrix_from_file(path);
    if (!cooMatrix) {
        return -1;
    }
    Vector *x, *y;
    x = Vector_new(cooMatrix->row_size);
    if (!x) {
        COOMatrix_free(cooMatrix);
        return -1;
    }
    y = Vector_new(cooMatrix->col_size);
    if (!y) {
        Vector_free(x);
        COOMatrix_free(cooMatrix);
        return -1;
    }
    Benchmark benchmark;
    CSRMatrix_SpMV(cooMatrix, x, y, 512, &benchmark);
    printf("%s, %s, %s, %f, %f\n", path, "CSR", "CUDA", benchmark.gpuTime / 512.0, compute_FLOPS(cooMatrix->num_non_zero_elements, benchmark.gpuTime / 512) / 1000000000);
    Vector_free(x);
    Vector_free(y);
    COOMatrix_free(cooMatrix);
    return 0;
}

int do_ELLMatrix_prod(char *path, struct dirent *entry, void *args) {
    (void) entry;
    (void) args;
    COOMatrix *cooMatrix = read_matrix_from_file(path);
    if (!cooMatrix) {
        return -1;
    }
    Vector *x, *y;
    x = Vector_new(cooMatrix->row_size);
    if (!x) {
        COOMatrix_free(cooMatrix);
        return -1;
    }
    y = Vector_new(cooMatrix->col_size);
    if (!y) {
        Vector_free(x);
        COOMatrix_free(cooMatrix);
        return -1;
    }
    Benchmark benchmark;
    ELLMatrix_SpMV(cooMatrix, x, y, 512, &benchmark);
    printf("%s, %s, %s, %f, %f\n", path, "ELL", "CUDA", benchmark.gpuTime / 512.0, compute_FLOPS(cooMatrix->num_non_zero_elements, benchmark.gpuTime / 512) / 1000000000);
    Vector_free(x);
    Vector_free(y);
    COOMatrix_free(cooMatrix);
    return 0;
}

int do_CSRMatrix_OMP_prod(char *path, struct dirent *entry, void *args) {
    (void) entry;
    (void) args;
    u_int64_t num_op = 512;
    COOMatrix *cooMatrix = read_matrix_from_file(path);
    if (!cooMatrix) {
        return -1;
    }
    Vector *x, *y;
    x = Vector_new(cooMatrix->row_size);
    if (!x) {
        COOMatrix_free(cooMatrix);
        return -1;
    }
    y = Vector_new(cooMatrix->col_size);
    if (!y) {
        Vector_free(x);
        COOMatrix_free(cooMatrix);
        return -1;
    }
    Benchmark benchmark;
    CSRMatrix_SpMV_cpu(cooMatrix, x, y, num_op, &benchmark);
    printf("%s, %s, %s, %f, %f\n", path, "CSR", "OMP", benchmark.cpuTime / (double) num_op, compute_FLOPS(cooMatrix->num_non_zero_elements, benchmark.cpuTime / num_op) / 1000000000);
    Vector_free(x);
    Vector_free(y);
    COOMatrix_free(cooMatrix);
    return 0;

}

int do_ELLMatrix_OMP_prod(char *path, struct dirent *entry, void *args) {
    (void) entry;
    (void) args;
    u_int64_t num_op = 512;
    COOMatrix *cooMatrix = read_matrix_from_file(path);
    if (!cooMatrix) {
        return -1;
    }
    Vector *x, *y;
    x = Vector_new(cooMatrix->row_size);
    if (!x) {
        COOMatrix_free(cooMatrix);
        return -1;
    }
    y = Vector_new(cooMatrix->col_size);
    if (!y) {
        Vector_free(x);
        COOMatrix_free(cooMatrix);
        return -1;
    }
    Benchmark benchmark;
    ELLMatrix_SpMV_cpu(cooMatrix, x, y, num_op, &benchmark);
    printf("%s, %s, %s, %f, %f\n", path, "ELL", "OMP", benchmark.cpuTime / (double) num_op, compute_FLOPS(cooMatrix->num_non_zero_elements, benchmark.cpuTime / num_op) / 1000000000);
    Vector_free(x);
    Vector_free(y);
    COOMatrix_free(cooMatrix);
    return 0;
}


int main(int argc, char *argv[]) {
    int ret;
    if (argc < 2) {
        return EXIT_FAILURE;
    }
    char *matrixDirPath = argv[1];
    printf("%s, %s, %s, %s, %s\n", "Matrix", "Format", "Device", "Time", "FLOPS");
    ret = forEachFile(matrixDirPath, do_CSRMatrix_prod, NULL);
    if (ret < 0) {
        return EXIT_FAILURE;
    }
    
    //printf("%s, %s, %s, %s, %s\n", "Matrix", "Format", "Device", "Time", "GFLOPS");
    ret = forEachFile(matrixDirPath, do_ELLMatrix_prod, NULL);
    if (ret < 0) {
        return EXIT_FAILURE;
    }

    //printf("%s, %s, %s, %s, %s\n", "Matrix", "Format", "Device", "Time", "GFLOPS");
    ret = forEachFile(matrixDirPath, do_CSRMatrix_OMP_prod, NULL);
    if (ret < 0) {
        return EXIT_FAILURE;
    }
    
    //  printf("%s, %s, %s, %s, %s\n", "Matrix", "Format", "Device", "Time", "GFLOPS");
    ret = forEachFile(matrixDirPath, do_ELLMatrix_OMP_prod, NULL);
    if (ret < 0) {
        return EXIT_FAILURE;
    }

    return 0;
}

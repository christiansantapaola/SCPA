//
// Created by 9669c on 24/03/2022.
//

#include <stdio.h>

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMV.h"

float TEST_MATRIX[4][4] = {11.0f, 12.0f, 0.0f, 0.0f,
                           0.0f, 22.0f, 23.0f, 0.0f,
                           0.0f, 0.0f, 33.0f, 0.0f,
                           0.0f, 0.0f, 43.0f, 44.0f};

const char *PROGRAM_NAME = "spmvCSR";

int main(int argc, char *argv[]) {
    if (argc < 2) {
        return 1;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror(argv[1]);
        return 1;
    }
    // COOMatrix cooMatrix = COOMatrix((float*)TEST_MATRIX, 4, 4);
    COOMatrix *cooMatrix = COOMatrix_new(file);
    if (file != stdin) {
        fclose(file);
    }
    CSRMatrix *csrMatrix = CSRMatrix_pinned_memory_new(cooMatrix);
    if (!csrMatrix) {
        perror("CSRMatrix_pinned_memory_new()");
        return EXIT_FAILURE;
    }
    Vector* X = Vector_pinned_memory_new(csrMatrix->col_size);
    if (!X) {
        perror("Vector_pinned_memory_new()");
        return EXIT_FAILURE;
    }
    Vector* Y = Vector_new(csrMatrix->row_size);
    if (!Y) {
        perror("Vector_new");
        return EXIT_FAILURE;
    }
    Vector* Z = Vector_pinned_memory_new(csrMatrix->row_size);
    if (!Z) {
        perror("Vector_pinned_memory_new");
        return EXIT_FAILURE;
    }
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    Vector_set(Z, 0.0f);
    SpMVResultCPU cpuResult;
    CSRMatrix_SpMV_CPU(csrMatrix, X, Y, &cpuResult);
    SpMVResultCUDA gpuResult;
    CSRMatrix_SpMV_GPU(csrMatrix, X, Z, &gpuResult);
    int success = Vector_equals(Y, Z);
    fprintf(stdout, "{\n");
    fprintf(stdout, "\"success\": %s,\n", (success) ? "true" : "false");
    fprintf(stdout, "\"MatrixInfo\": ");
    CSRMatrix_infoOutAsJSON(csrMatrix, stdout);
    fprintf(stdout, ",\n");
    fprintf(stdout, "\"CPUresult\": ");
    SpMVResultCPU_outAsJSON(&cpuResult, stdout);
    fprintf(stdout, ",\n");
    fprintf(stdout, "\"GPUresult\": ");
    SpMVResultCUDA_outAsJSON(&gpuResult, stdout);
    fprintf(stdout, "\n}\n");
    Vector_pinned_memory_free(Z);
    Vector_free(Y);
    Vector_pinned_memory_free(X);
    CSRMatrix_pinned_memory_free(csrMatrix);
    COOMatrix_free(cooMatrix);
    return EXIT_SUCCESS;
}
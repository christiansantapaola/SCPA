//
// Created by 9669c on 24/03/2022.
//

#include <stdio.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "Vector.h"
#include "SpMV.h"

#define PROGRAM_NAME "spmvCSRCuda"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return 1;
    }
    MTXParser *mtxParser = MTXParser_new(argv[1]);
    if (!mtxParser) {
        perror("MTXParser_new()");
        exit(EXIT_FAILURE);
    }
    COOMatrix *cooMatrix = MTXParser_parse(mtxParser);
    if (!cooMatrix) {
        perror("MTXParser_parse():");
        MTXParser_free(mtxParser);
        return EXIT_FAILURE;
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
    if (success) {
        fprintf(stdout, ",\n");
        fprintf(stdout, "\"CPUresult\": ");
        SpMVResultCPU_outAsJSON(&cpuResult, stdout);
        fprintf(stdout, ",\n");
        fprintf(stdout, "\"GPUresult\": ");
        SpMVResultCUDA_outAsJSON(&gpuResult, stdout);
    }
    fprintf(stdout, "\n}\n");
    Vector_pinned_memory_free(Z);
    Vector_free(Y);
    Vector_pinned_memory_free(X);
    CSRMatrix_pinned_memory_free(csrMatrix);
    COOMatrix_free(cooMatrix);
    return EXIT_SUCCESS;
}
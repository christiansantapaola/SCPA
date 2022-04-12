#include <stdio.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"

#define PROGRAM_NAME "spmvELLCuda"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return EXIT_FAILURE;
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
    CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
    ELLMatrix *ellMatrix = ELLMatrix_pinned_memory_new(csrMatrix);
    Vector* X = Vector_pinned_memory_new(ellMatrix->col_size);
    Vector* Y = Vector_new(ellMatrix->row_size);
    Vector* Z = Vector_pinned_memory_new(ellMatrix->row_size);
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    Vector_set(Z, 0.0f);
    SpMVResultCPU cpuResult;
    ELLMatrix_SpMV_CPU(ellMatrix, X, Y, &cpuResult);
    SpMVResultCUDA gpuResult;
    ELLMatrix_transpose(ellMatrix);
    ELLMatrix_SpMV_GPU(ellMatrix, X, Z, &gpuResult);
    int success = Vector_equals(Y, Z);
    fprintf(stdout, "{\n");
    fprintf(stdout, "\"success\": %s,\n", (success) ? "true" : "false");
    fprintf(stdout, "\"MatrixInfo\": ");
    ELLMatrix_infoOutAsJSON(ellMatrix, stdout);
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
    ELLMatrix_pinned_memory_free(ellMatrix);
    CSRMatrix_free(csrMatrix);
    COOMatrix_free(cooMatrix);
    return EXIT_SUCCESS;
}
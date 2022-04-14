#include <stdio.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"

#define PROGRAM_NAME "spmvELLOpenMP"

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
    Vector* X = Vector_new(cooMatrix->col_size);
    Vector* Y = Vector_new(cooMatrix->row_size);
    Vector* Z = Vector_new(cooMatrix->row_size);
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    Vector_set(Z, 0.0f);
    COOMatrix *first, *second;
    first = COOMatrix_new();
    second = COOMatrix_new();
    int noSplit = COOMatrix_split(cooMatrix, first, second, 64);
    if (noSplit == -1) {
        fprintf(stderr, "COOMatrix_split failed!");
        exit(EXIT_FAILURE);
    }
    if (noSplit) {
        ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(cooMatrix);
        SpMVResultCPU cpuResult, openmpResult;
        ELLMatrix_SpMV_CPU(ellMatrix, X, Y, &cpuResult);
        ELLMatrix_SpMV_OPENMP(ellMatrix, X, Z, &openmpResult);
        int success = Vector_equals(Y, Z);
        fprintf(stdout, "{\n");
        fprintf(stdout, "\"success\": %s,\n", (success) ? "true" : "false");
        fprintf(stdout, "\"MatrixInfo\": ");
        ELLMatrix_infoOutAsJSON(ellMatrix, stdout);
        fprintf(stdout, ",\n");
        fprintf(stdout, "\"CPUresult\": ");
        SpMVResultCPU_outAsJSON(&cpuResult, stdout);
        fprintf(stdout, ",\n");
        fprintf(stdout, "\"OpenMPResult\": ");
        SpMVResultCUDA_outAsJSON(&openmpResult, stdout);
        fprintf(stdout, "\n}\n");
        ELLMatrix_free(ellMatrix);
    } else {
        ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(first);
        SpMVResultCPU cpuResult, openmpELL, openmpCOO;
        COOMatrix_SpMV_CPU(cooMatrix, X, Y, &cpuResult);
        ELLMatrix_SpMV_OPENMP(ellMatrix, X, Z, &openmpELL);
        COOMatrix_SpMV_OPENMP(second, X, Z, &openmpCOO);
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
        fprintf(stdout, ",\n");
        fprintf(stdout, "\"GPUCOOResult\": ");
        SpMVResultCUDA_outAsJSON(&gpuCOOResult, stdout);
        fprintf(stdout, "\n}\n");
        ELLMatrix_free(ellMatrix);
    }
    COOMatrix_free(first);
    COOMatrix_free(second);
    Vector_free(Z);
    Vector_free(Y);
    Vector_free(X);
    COOMatrix_free(cooMatrix);
    MTXParser_free(mtxParser);
    return EXIT_SUCCESS;
}
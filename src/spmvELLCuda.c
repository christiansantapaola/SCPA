#include <stdio.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"

#define PROGRAM_NAME "spmvELLCuda"
#define WPM

void spmvWithPinnedMemory(char *mtx) {
    MTXParser *mtxParser = MTXParser_new(mtx);
    if (!mtxParser) {
        perror("MTXParser_new()");
        exit(EXIT_FAILURE);
    }
    COOMatrix *cooMatrix = MTXParser_parse(mtxParser);
    if (!cooMatrix) {
        perror("MTXParser_parse():");
        MTXParser_free(mtxParser);
        exit(EXIT_FAILURE);
    }
    Vector* X = Vector_new_wpm(cooMatrix->col_size);
    Vector* Y = Vector_new(cooMatrix->row_size);
    Vector* Z = Vector_new_wpm(cooMatrix->row_size);
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    Vector_set(Z, 0.0f);
    COOMatrix *first, *second;
    first = COOMatrix_new();
    second = COOMatrix_new();
    int noSplit = COOMatrix_split_wpm(cooMatrix, first, second, 64);
    if (noSplit == -1) {
        fprintf(stderr, "COOMatrix_split failed!");
        exit(EXIT_FAILURE);
    }
    if (noSplit) {
        ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO_wpm(cooMatrix);
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
        ELLMatrix_free_wpm(ellMatrix);
    } else {
        ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO_wpm(first);
        SpMVResultCPU cpuResult;
        COOMatrix_SpMV_CPU(cooMatrix, X, Y, &cpuResult);
        SpMVResultCUDA gpuResult;
        SpMVResultCUDA gpuCOOResult;
        ELLMatrix_transpose(ellMatrix);
        ELLMatrix_SpMV_GPU(ellMatrix, X, Z, &gpuResult);
        COOMatrix_SpMV_GPU(second, X, Z, &gpuCOOResult);
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
        ELLMatrix_free_wpm(ellMatrix);
    }
    COOMatrix_free(first);
    COOMatrix_free_wpm(second);
    Vector_free_wpm(Z);
    Vector_free(Y);
    Vector_free_wpm(X);
    COOMatrix_free(cooMatrix);
    MTXParser_free(mtxParser);
}

void spmvWithoutPinnedMemory(char *mtx) {
    MTXParser *mtxParser = MTXParser_new(mtx);
    if (!mtxParser) {
        perror("MTXParser_new()");
        exit(EXIT_FAILURE);
    }
    COOMatrix *cooMatrix = MTXParser_parse(mtxParser);
    if (!cooMatrix) {
        perror("MTXParser_parse():");
        MTXParser_free(mtxParser);
        exit(EXIT_FAILURE);
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
        ELLMatrix_free(ellMatrix);
    } else {
        ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(first);
        SpMVResultCPU cpuResult;
        COOMatrix_SpMV_CPU(cooMatrix, X, Y, &cpuResult);
        SpMVResultCUDA gpuResult;
        SpMVResultCUDA gpuCOOResult;
        ELLMatrix_transpose(ellMatrix);
        ELLMatrix_SpMV_GPU(ellMatrix, X, Z, &gpuResult);
        COOMatrix_SpMV_GPU(second, X, Z, &gpuCOOResult);
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
    COOMatrix_free_wpm(second);
    Vector_free(Z);
    Vector_free(Y);
    Vector_free(X);
    COOMatrix_free(cooMatrix);
    MTXParser_free(mtxParser);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
#ifdef WPM
    spmvWithPinnedMemory(argv[1]);
#else
    spmvWithoutPinnedMemory(argv[1]);
#endif
    return EXIT_SUCCESS;
}
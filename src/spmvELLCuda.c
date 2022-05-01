#include <stdio.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"

#define PROGRAM_NAME "spmvELLCuda"
#define MATRIX_SPLIT_THRESHOLD 32

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
        exit(EXIT_FAILURE);
    }
    Vector* X = Vector_new_wpm(cooMatrix->col_size);
    Vector* Y = Vector_new_wpm(cooMatrix->row_size);
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    COOMatrix *lower, *higher;
    lower = COOMatrix_new();
    higher = COOMatrix_new();
    int noSplit = COOMatrix_split(cooMatrix, lower, higher, MATRIX_SPLIT_THRESHOLD);
    if (noSplit == -1) {
        fprintf(stderr, "COOMatrix_split failed!");
        exit(EXIT_FAILURE);
    }
    SpMVResultCPU cpuResult;
    SpMVResultCUDA gpuResult;
    if (noSplit) {
        ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO_wpm(cooMatrix);
        ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
        ELLMatrix_free_wpm(h_ellMatrix);
        Vector *d_x = Vector_to_CUDA(X);
        Vector *d_y = Vector_to_CUDA(Y);
        ELLMatrix_SpMV_CUDA(d_ellMatrix, d_x, d_y, &gpuResult);
        Vector *Z = Vector_from_CUDA(d_y);
        Vector_free_CUDA(d_y);
        Vector_free_CUDA(d_x);
        Vector_free(Z);
    } else {
        ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO_wpm(lower);
        ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
        ELLCOOMatrix_SpMV_CUDA(d_ellMatrix, higher, X, Y, &gpuResult);
        ELLMatrix_free_wpm(h_ellMatrix);
    }
    fprintf(stdout, "{\n");
    fprintf(stdout, "\"split\": %s,\n", "true");
    fprintf(stdout, "\"MatrixInfo\": ");
    COOMatrix_infoOutAsJSON(cooMatrix, stdout);
    fprintf(stdout, ",\n");
    fprintf(stdout, "\"CPUresult\": ");
    SpMVResultCPU_outAsJSON(&cpuResult, stdout);
    fprintf(stdout, ",\n");
    fprintf(stdout, "\"GPUresult\": ");
    SpMVResultCUDA_outAsJSON(&gpuResult, stdout);
    fprintf(stdout, "\n}\n");

    COOMatrix_free(lower);
    COOMatrix_free_wpm(higher);
    Vector_free_wpm(Y);
    Vector_free_wpm(X);
    COOMatrix_free(cooMatrix);
    MTXParser_free(mtxParser);
    return EXIT_SUCCESS;
}
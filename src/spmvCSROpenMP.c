//
// Created by 9669c on 24/03/2022.
//

#include <stdio.h>
#include "MTXParser.h"
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
        perror("fopen()");
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
    CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
    Vector* X = Vector_new(csrMatrix->col_size);
    Vector* Y = Vector_new(csrMatrix->row_size);
    Vector* Z = Vector_new(csrMatrix->row_size);
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    Vector_set(Z, 0.0f);
    SpMVResultCPU cpuResult;
    CSRMatrix_SpMV_CPU(csrMatrix, X, Y, &cpuResult);
    SpMVResultCPU openMPResult;
    CSRMatrix_SpMV_OPENMP(csrMatrix, X, Z, &openMPResult);
    int success = Vector_equals(Y, Z);
    fprintf(stdout, "{\n");
    fprintf(stdout, "\"success\": %s,\n", (success) ? "true" : "false");
    fprintf(stdout, "\"CPUresult\": ");
    SpMVResultCPU_outAsJSON(&cpuResult, stdout);
    fprintf(stdout, ",\n");
    fprintf(stdout, "\"openMPResult\": ");
    SpMVResultCPU_outAsJSON(&openMPResult, stdout);
    fprintf(stdout, "\n}\n");
    Vector_free(Z);
    Vector_free(Y);
    Vector_free(X);
    CSRMatrix_free(csrMatrix);
    COOMatrix_free(cooMatrix);
    return EXIT_SUCCESS;
}
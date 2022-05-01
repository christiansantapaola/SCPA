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
    CSRMatrix *csrMatrix = CSRMatrix_new_wpm(cooMatrix);
    if (!csrMatrix) {
        perror("CSRMatrix_new_wpm()");
        return EXIT_FAILURE;
    }
    Vector* X = Vector_new_wpm(csrMatrix->col_size);
    if (!X) {
        perror("Vector_new_wpm()");
        return EXIT_FAILURE;
    }
    Vector* Y = Vector_new_wpm(csrMatrix->row_size);
    if (!Y) {
        perror("Vector_new");
        return EXIT_FAILURE;
    }
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    SpMVResultCPU cpuResult = {0};
    SpMVResultCUDA gpuResult = {0};
    CSRMatrix *d_csrMatrix = CSRMatrix_to_CUDA(csrMatrix);
    Vector *d_x = Vector_to_CUDA(X);
    Vector *d_y = Vector_to_CUDA(Y);
    COOMatrix_SpMV(cooMatrix, X, Y, &cpuResult);
    CSRMatrix_SpMV_CUDA(d_csrMatrix, d_x, d_y, &gpuResult);
    Vector *Z = Vector_from_CUDA(d_y);
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
    Vector_free_CUDA(d_x);
    Vector_free_CUDA(d_y);
    Vector_free(Z);
    Vector_free_wpm(Y);
    Vector_free_wpm(X);
    CSRMatrix_free_wpm(csrMatrix);
    COOMatrix_free(cooMatrix);
    return EXIT_SUCCESS;
}
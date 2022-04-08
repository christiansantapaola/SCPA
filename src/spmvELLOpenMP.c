#include <stdio.h>

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "SpMV.h"

float TEST_MATRIX[4][4] = {11.0f, 12.0f, 0.0f, 0.0f,
                           0.0f, 22.0f, 23.0f, 0.0f,
                           0.0f, 0.0f, 33.0f, 0.0f,
                           0.0f, 0.0f, 43.0f, 44.0f};

const char *PROGRAM_NAME = "spmvELL";

int main(int argc, char *argv[]) {
    if (argc < 2) {
        return 1;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("fopen()");
        return 1;
    }
    // COOMatrix cooMatrix = COOMatrix((float*)TEST_MATRIX, 4, 4);
    COOMatrix *cooMatrix = COOMatrix_new(file);
    if (file != stdin) {
        fclose(file);
    }
    CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
    ELLMatrix *ellMatrix = ELLMatrix_new(csrMatrix);
    Vector* X = Vector_new(ellMatrix->col_size);
    Vector* Y = Vector_new(ellMatrix->row_size);
    Vector* Z = Vector_new(ellMatrix->row_size);
    Vector_set(X, 1.0f);
    Vector_set(Y, 0.0f);
    Vector_set(Z, 0.0f);
    SpMVResultCPU cpuResult;
    ELLMatrix_SpMV_CPU(ellMatrix, X, Y, &cpuResult);
    SpMVResultCPU openmpResult;
    ELLMatrix_SpMV_OPENMP(ellMatrix, X, Z, &openmpResult);
    int success = Vector_equals(Y, Z);
    fprintf(stdout, "{\n");
    fprintf(stdout, "\"success\": %s,\n", (success) ? "true" : "false");
    if (!success) {
        fprintf(stdout, "Y: ");
        Vector_outAsJSON(Y, stdout);
        putc(',', stdout);
        putc('\n', stdout);
        fprintf(stdout, "Z: ");
        Vector_outAsJSON(Z, stdout);
    } else {
        fprintf(stdout, "\"CPUresult\": ");
        SpMVResultCPU_outAsJSON(&cpuResult, stdout);
        fprintf(stdout, ",\n");
        fprintf(stdout, "\"OpenMPResult\": ");
        SpMVResultCPU_outAsJSON(&openmpResult, stdout);
    }
    fprintf(stdout, "\n}\n");
    Vector_free(Z);
    Vector_free(Y);
    Vector_free(X);
    ELLMatrix_free(ellMatrix);
    CSRMatrix_free(csrMatrix);
    COOMatrix_free(cooMatrix);
    return EXIT_SUCCESS;
}
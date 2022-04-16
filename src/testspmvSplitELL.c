#include <stdio.h>
#include <stdlib.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "ELLMatrix.h"
#include "Histogram.h"
#include "Vector.h"
#include "SpMV.h"

#define PROGRAM_NAME "spmvCOOSplit"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "%s file.mtx\n", PROGRAM_NAME);
        exit(EXIT_FAILURE);
    }
    MTXParser *parser = MTXParser_new(argv[1]);
    if (!parser) {
        perror(argv[1]);
        exit(EXIT_FAILURE);
    }
    COOMatrix *matrix = MTXParser_parse(parser);
    if (!matrix) {
        exit(EXIT_FAILURE);
    }
    Vector *X = Vector_new(matrix->row_size);
    if (!X) {
        fprintf(stderr, "Vector_new_wpm(%lu)", matrix->row_size);
        perror("");
        exit(EXIT_FAILURE);
    }
    Vector_set(X, 1.0f);
    Vector *Y = Vector_new(matrix->col_size);
    if (!Y) {
        fprintf(stderr, "Vector_new_wpm(%lu)", matrix->col_size);
        perror("");
        exit(EXIT_FAILURE);
    }
    Vector_set(Y, 0.0f);
    Vector *Z = Vector_new(matrix->col_size);
    if (!Z) {
        fprintf(stderr, "Vector_(%lu)", matrix->col_size);
        perror("");
        exit(EXIT_FAILURE);
    }
    Vector_set(Z, 0.0f);

    SpMVResultCPU cpu;
    SpMVResultCUDA cpu1, cpu2;
    COOMatrix *lower, *higher;
    lower = COOMatrix_new();
    higher = COOMatrix_new();
    int noSplit = COOMatrix_split(matrix, lower, higher, 64);
    if (noSplit == -1) {
        fprintf(stderr, "FAIL\n");
        exit(EXIT_FAILURE);
    }
    if (noSplit) {
        fprintf(stderr, "NOSPLIT\n");
        exit(EXIT_FAILURE);
    }
    ELLMatrix *ellLower = ELLMatrix_new_fromCOO(lower);
    //COOMatrix_SpMV_CPU(lower, X, Y, &cpu);
    //ELLMatrix_SpMV_CPU(ellLower, X, Y, &cpu);
    // COOMatrix_SpMV_CPU(matrix, X, Y, &cpu);
    ELLMatrix_transpose(ellLower);
    ELLMatrixHyb_SpMV_GPU(ellLower, higher, X, Y,&cpu2);
    ELLMatrixHyb_SpMV_GPU(ellLower, higher, X, Z,&cpu1);
    int success = Vector_equals(Y, Z);
    fprintf(stdout, "{\n");
    fprintf(stdout, "\"success\": %s,\n", (success) ? "true" : "false");
    fprintf(stdout, "\"MatrixInfo\": ");
    COOMatrix_infoOutAsJSON(matrix, stdout);
    fprintf(stdout, "\n}\n");
    Vector_free(Z);
    Vector_free(Y);
    Vector_free(X);
    COOMatrix_free(matrix);
    return 0;
}
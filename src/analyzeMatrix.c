#include <stdio.h>
#include <stdlib.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "Histogram.h"
#include "Vector.h"
#include "SpMV.h"

#define PROGRAM_NAME "analyzeMatrix"

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
        fprintf(stderr, "Vector_pinned_memory_new(%lu)", matrix->row_size);
        perror("");
        exit(EXIT_FAILURE);
    }
    Vector_set(X, 1.0f);
    Vector *Y = Vector_new(matrix->col_size);
    if (!Y) {
        fprintf(stderr, "Vector_pinned_memory_new(%lu)", matrix->col_size);
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
    Vector *U = Vector_new(matrix->col_size);
    if (!U) {
        fprintf(stderr, "Vector_(%lu)", matrix->col_size);
        perror("");
        exit(EXIT_FAILURE);
    }
    Vector_set(U, 0.0f);

    COOMatrix first, second;
    int ret = COOMatrix_split(matrix, &first, &second, 5);
    if (ret == -1) {
        fprintf(stderr, "fail\n");
        return EXIT_FAILURE;
    }
    //COOMatrix_outAsJSON(matrix, stdout);
    //COOMatrix_outAsJSON(&first, stdout);
    //COOMatrix_outAsJSON(&second, stdout);
    SpMVResultCPU cpu1, cpu2, cpu3;
    COOMatrix_SpMV_CPU(matrix, X, Y, &cpu1);
    //COOMatrix_SpMV_CPU(&first, X, Z, &cpu2);
    //COOMatrix_SpMV_CPU(&second, X, Z, &cpu3);
    SpMVResultCUDA r1, r2, r3, r4;
    CSRMatrix *m1, *m2;
    ELLMatrix *m3;
    m2 = CSRMatrix_new(&second);
    m3 = ELLMatrix_new_fromCOO(&first);
    ELLMatrix_outAsJSON(m3, stdout);
    ELLMatrix_transpose(m3);
    ELLMatrix_outAsJSON(m3, stdout);
    ELLMatrix_SpMV_GPU(m3, X, Z, &r1);
    CSRMatrix_SpMV_GPU(m2, X, Z, &r2);

    fprintf(stdout, "\"success\": \"%s\"\n", (Vector_equals(Y, Z)) ? "True" : "False");

    return 0;
}
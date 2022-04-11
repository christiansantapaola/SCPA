#include <stdio.h>
#include <dirent.h>
#include<string.h>
#include<stdlib.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "Vector.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "SpMV.h"


const char *PROGRAM_NAME = "spmvCSR_stat";

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s dir [output.json]\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
    DIR *dir = opendir(argv[1]);
    if (!dir) {
        perror(argv[1]);
        fprintf(stderr, "USAGE: %s dir\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
    struct dirent *entry;
    FILE *out = (argc >= 3) ? fopen(argv[2], "r") : stdout;
    if (!out) {
        perror(argv[2]);
        closedir(dir);
        return EXIT_FAILURE;
    }
    fprintf(out, "{\n");
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) {
            continue;
        }
        MTXParser *mtxParser = MTXParser_new(entry->d_name);
        if (!mtxParser) {
            fprintf(stderr, "MTXParser_new(%p) failed\n", entry->d_name);
            exit(EXIT_FAILURE);
        }
        COOMatrix *cooMatrix = MTXParser_parse(mtxParser);
        if (!cooMatrix) {
            fprintf(stderr, "MTXParser_parser(%p) failed\n", mtxParser);
            exit(EXIT_FAILURE);
        }
        Vector *X = Vector_pinned_memory_new(cooMatrix->row_size);
        Vector_set(X, 1.0f);
        Vector *Y = Vector_pinned_memory_new(cooMatrix->col_size);
        Vector_set(Y, 0.0f);
        Vector *Z = Vector_new(cooMatrix->col_size);
        Vector_set(Z, 0.0f);
        Vector *U = Vector_new(cooMatrix->col_size);
        Vector_set(U, 0.0f);
        CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
        ELLMatrix *ellMatrix = ELLMatrix_pinned_memory_new(csrMatrix);
        SpMVResultCPU cpuResult;
        SpMVResultCUDA gpuResult;
        SpMVResultCPU openmpResult;
        ELLMatrix_SpMV_GPU(ellMatrix, X, Y, &gpuResult);
        ELLMatrix_SpMV_CPU(ellMatrix, X, Z, &cpuResult);
        ELLMatrix_SpMV_OPENMP(ellMatrix, X, U, &openmpResult);
        int successGPU = Vector_equals(Y, Z);
        int successOpenMP = Vector_equals(Z, U);
        fprintf(out, "{\n");
        fprintf(out, "\"matrix: \" %s,\n", entry->d_name);
        fprintf(out, "\"successGPU\": %s,\n", (successGPU) ? "true" : "false");
        fprintf(out, "\"successOpenMP\": %s,\n", (successOpenMP) ? "true" : "false");
        fprintf(out, "\"MatrixInfo\": ");
        CSRMatrix_infoOutAsJSON(csrMatrix, out);
        fprintf(out, ",\n");
        fprintf(out, "\"CPUresult\": ");
        SpMVResultCPU_outAsJSON(&cpuResult, out);
        fprintf(out, ",\n");
        fprintf(out, "\"GPUresult\": ");
        SpMVResultCUDA_outAsJSON(&gpuResult, out);
        fprintf(out, ",\n");
        fprintf(out, "\"OpenMPresult\": ");
        SpMVResultCPU_outAsJSON(&openmpResult, out);
        fprintf(out, "\n},\n");
        ELLMatrix_pinned_memory_free(ellMatrix);
        CSRMatrix_free(csrMatrix);
        Vector_pinned_memory_free(X);
        Vector_pinned_memory_free(Y);
        Vector_free(Z);
        COOMatrix_free(cooMatrix);
        MTXParser_free(mtxParser);
    }
    fprintf(out, "{}\n}\n");
    return EXIT_SUCCESS;
}
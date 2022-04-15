#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "Vector.h"
#include "CSRMatrix.h"
#include "SpMV.h"

#define PROGRAM_NAME "spmvCSR"

void print_status_bar(int used, int total,char *file) {
    fprintf(stderr, "\33[2K\r[");
    for (int i = 0; i < used; i++) {
        fprintf(stderr, "#");
    }
    for (int i = used; i < total; i++) {
        fprintf(stderr, "-");
    }
    fprintf(stderr, "] %.2f %s", (double) used / (double) total, file);
}

int count_directory(const char *dirpath) {
    int count = 0;
    struct dirent *entry;
    DIR *dir = opendir(dirpath);
    if (!dir) {
        perror(dirpath);
        fprintf(stderr, "USAGE: %s dir\n", PROGRAM_NAME);
        return -1;
    }
    while ((entry = readdir(dir)) != NULL) {
        count++;
    }
    closedir(dir);
    return count;
}

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
    int numDir = count_directory(argv[1]) - 2;
    struct dirent *entry;
    FILE *out = (argc >= 3) ? fopen(argv[2], "wb+") : stdout;
    if (!out) {
        perror(argv[2]);
        closedir(dir);
        return EXIT_FAILURE;
    }
    fprintf(out, "{ \"CSRResult\": [\n");
    char absolutePath [PATH_MAX+1];
    chdir(argv[1]);
    int fileProcessed = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type != DT_REG) {
            continue;
        }
        print_status_bar(fileProcessed, numDir, entry->d_name);
        fileProcessed++;
        memset(absolutePath, 0, PATH_MAX + 1);
        char *ptr = realpath(entry->d_name, absolutePath);
        if (!ptr) {
            perror(entry->d_name);
            exit(EXIT_FAILURE);
        }
        MTXParser *mtxParser = MTXParser_new(absolutePath);
        if (!mtxParser) {
            fprintf(stderr, "MTXParser_new(\"%s\") failed\n", absolutePath);
            exit(EXIT_FAILURE);
        }
        COOMatrix *cooMatrix = MTXParser_parse(mtxParser);
        if (!cooMatrix) {
            fprintf(stderr, "MTXParser_parser(%p) failed\n", mtxParser);
            exit(EXIT_FAILURE);
        }
        Vector *X = Vector_new_wpm(cooMatrix->row_size);
        if (!X) {
            fprintf(stderr, "Vector_new_wpm(%lu)", cooMatrix->row_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(X, 1.0f);
        Vector *Y = Vector_new_wpm(cooMatrix->col_size);
        if (!Y) {
            fprintf(stderr, "Vector_new_wpm(%lu)", cooMatrix->col_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(Y, 0.0f);
        Vector *Z = Vector_new(cooMatrix->col_size);
        if (!Z) {
            fprintf(stderr, "Vector_(%lu)", cooMatrix->col_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(Z, 0.0f);
        Vector *U = Vector_new(cooMatrix->col_size);
        if (!U) {
            fprintf(stderr, "Vector_(%lu)", cooMatrix->col_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(U, 0.0f);
        CSRMatrix *csrMatrix = CSRMatrix_new_wpm(cooMatrix);
        if (!csrMatrix) {
            perror("CSRMatrix_new_wpm()");
            exit(EXIT_FAILURE);

        }
        SpMVResultCPU cpuResult;
        SpMVResultCUDA gpuResult;
        SpMVResultCPU openmpResult;
        CSRMatrix_SpMV_GPU_wpm(csrMatrix, X, Y, &gpuResult);
        CSRMatrix_SpMV_CPU(csrMatrix, X, Z, &cpuResult);
        CSRMatrix_SpMV_OPENMP(csrMatrix, X, U, &openmpResult);
        int successGPU = Vector_equals(Z, Y);
        int successOpenMP = Vector_equals(Z, U);
        fprintf(out, "{\n");
        fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
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
        CSRMatrix_free_wpm(csrMatrix);
        Vector_free_wpm(X);
        Vector_free_wpm(Y);
        Vector_free(Z);
        COOMatrix_free(cooMatrix);
        MTXParser_free(mtxParser);
    }
    fprintf(out, "{}\n]}\n");
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "Vector.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "SpMV.h"


#define PROGRAM_NAME "spmvELL"
#define MATRIX_SPLIT_THRESHOLD 32

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
    fprintf(out, "{ \"ELLResult\": [\n");
    char absolutePath[PATH_MAX + 1];
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
        COOMatrix *lower, *higher;
        lower = COOMatrix_new();
        if (!lower) {
            perror("COOMatrix_new()");
            exit(EXIT_FAILURE);
        }
        higher = COOMatrix_new();
        if (!higher) {
            perror("COOMatrix_new()");
            exit(EXIT_FAILURE);
        }
        int noSplit = COOMatrix_split_wpm(cooMatrix, lower, higher, MATRIX_SPLIT_THRESHOLD);
        if (noSplit == -1) {
            fprintf(stderr, "error in COOMatrix_split:\n");
            exit(EXIT_FAILURE);
        }
        if (noSplit) {
            SpMVResultCPU cpuResult;
            SpMVResultCUDA gpuResult;
            ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO_wpm(cooMatrix);
            if (!ellMatrix) {
                perror("ELLMatrix_new()");
                exit(EXIT_FAILURE);
            }
            COOMatrix_SpMV_CPU(cooMatrix, X, Z, &cpuResult);
            ELLMatrix_transpose(ellMatrix);
            ELLMatrix_SpMV_GPU_wpm(ellMatrix, X, Y, &gpuResult);
            int successGPU = Vector_equals(Y, Z);
            fprintf(out, "{\n");
            fprintf(out, "\"matrix\": \"%s\",\n", entry->d_name);
            fprintf(out, "\"successGPU\": %s,\n", (successGPU) ? "true" : "false");
            fprintf(out, "\"split\": %s,\n", "false");
            fprintf(out, "\"MatrixInfo\": ");
            COOMatrix_infoOutAsJSON(cooMatrix, out);
            fprintf(out, ",\n");
            fprintf(out, "\"CPUresult\": ");
            SpMVResultCPU_outAsJSON(&cpuResult, out);
            fprintf(out, ",\n");
            fprintf(out, "\"GPUresult\": ");
            SpMVResultCUDA_outAsJSON(&gpuResult, out);
            fprintf(out, "\n},\n");
            ELLMatrix_free_wpm(ellMatrix);
        } else {
            SpMVResultCPU cpuResult;
            SpMVResultCUDA gpuResult;
            ELLMatrix *ellLower = ELLMatrix_new_fromCOO_wpm(lower);
            if (!ellLower) {
                perror("ELLMatrix_new()");
                exit(EXIT_FAILURE);
            }
            COOMatrix_SpMV_CPU(cooMatrix, X, Z, &cpuResult);
            ELLMatrix_transpose(ellLower);
            ELLMatrixHyb_SpMV_GPU_wpm(ellLower, higher, X, Z, &gpuResult);
            int successGPU = Vector_equals(Y, Z);
            fprintf(out, "{\n");
            fprintf(out, "\"matrix\": \"%s\",\n", entry->d_name);
            fprintf(out, "\"successGPU\": %s,\n", (successGPU) ? "true" : "false");
            fprintf(out, "\"split\": %s,\n", "true");
            fprintf(out, "\"MatrixInfo\": ");
            COOMatrix_infoOutAsJSON(cooMatrix, out);
            fprintf(out, ",\n");
            fprintf(out, "\"CPUresult\": ");
            SpMVResultCPU_outAsJSON(&cpuResult, out);
            fprintf(out, ",\n");
            fprintf(out, "\"GPUresult\": ");
            SpMVResultCUDA_outAsJSON(&gpuResult, out);
            fprintf(out, "\n},\n");
            ELLMatrix_free_wpm(ellLower);
        }
        Vector_free_wpm(X);
        Vector_free_wpm(Y);
        Vector_free(Z);
        COOMatrix_free(lower);
        COOMatrix_free_wpm(higher);
        COOMatrix_free(cooMatrix);
        MTXParser_free(mtxParser);
    }
    fprintf(out, "{}\n]}\n");
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
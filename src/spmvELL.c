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
#define MATRIX_SPLIT_THRESHOLD 64

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
        Vector *X = Vector_pinned_memory_new(cooMatrix->row_size);
        if (!X) {
            fprintf(stderr, "Vector_pinned_memory_new(%lu)", cooMatrix->row_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(X, 1.0f);
        Vector *Y = Vector_pinned_memory_new(cooMatrix->col_size);
        if (!Y) {
            fprintf(stderr, "Vector_pinned_memory_new(%lu)", cooMatrix->col_size);
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
        COOMatrix lower, higher;
        int ret = COOMatrix_split(cooMatrix, &lower, &higher, MATRIX_SPLIT_THRESHOLD);
        if (ret == -1) {
            fprintf(stderr, "error in COOMatrix_split:\n");
            exit(EXIT_FAILURE);
        }
        if (ret) {
            ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(cooMatrix);
            if (!ellMatrix) {
                perror("ELLMatrix_new()");
                exit(EXIT_FAILURE);
            }
            SpMVResultCPU cpuResult;
            SpMVResultCUDA gpuResult;
            SpMVResultCPU openmpResult;
            ELLMatrix_SpMV_GPU(ellMatrix, X, Y, &gpuResult);
            ELLMatrix_SpMV_CPU(ellMatrix, X, Z, &cpuResult);
            ELLMatrix_SpMV_OPENMP(ellMatrix, X, U, &openmpResult);
            int successGPU = Vector_equals(Y, Z);
            int successOpenMP = Vector_equals(Z, U);
            fprintf(out, "{\n");
            fprintf(out, "\"matrix\": \"%s\",\n", entry->d_name);
            fprintf(out, "\"successGPU\": %s,\n", (successGPU) ? "true" : "false");
            fprintf(out, "\"successOpenMP\": %s,\n", (successOpenMP) ? "true" : "false");
            fprintf(out, "\"MatrixInfo\": ");
            fprintf(out, ",\n");
            fprintf(out, "\"CPUresult\": ");
            SpMVResultCPU_outAsJSON(&cpuResult, out);
            fprintf(out, ",\n");
            fprintf(out, "\"GPUresult\": ");
            SpMVResultCUDA_outAsJSON(&gpuResult, out);
            fprintf(out, ",\n");
            fprintf(out, "\"GPUCSRresult\": ");
            fprintf(out, ",\n");
            fprintf(out, "\"OpenMPresult\": ");
            SpMVResultCPU_outAsJSON(&openmpResult, out);
            fprintf(out, "\n},\n");
            ELLMatrix_free(ellMatrix);
            Vector_pinned_memory_free(X);
            Vector_pinned_memory_free(Y);
            Vector_free(Z);
            Vector_free(U);
            COOMatrix_free(cooMatrix);
            MTXParser_free(mtxParser);
            fprintf(out, "{}\n]}\n");
            print_status_bar(numDir, numDir, "");
            fprintf(stderr, "\n");
        } else {
            ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(&lower);
            if (!ellMatrix) {
                perror("ELLMatrix_new()");
                exit(EXIT_FAILURE);
            }
            CSRMatrix *csrMatrix = CSRMatrix_new(&higher);
            if (!csrMatrix) {
                perror("CSRMatrix_new()");
                exit(EXIT_FAILURE);
            }
            SpMVResultCPU cpuResult;
            SpMVResultCUDA gpuResult;
            SpMVResultCUDA gpuCsrResult;
            SpMVResultCPU openmpResult;
            ELLMatrix_SpMV_GPU(ellMatrix, X, Y, &gpuResult);
            CSRMatrix_SpMV_GPU(csrMatrix, X, Y, &gpuCsrResult);
            ELLMatrix_SpMV_CPU(ellMatrix, X, Z, &cpuResult);
            ELLMatrix_SpMV_OPENMP(ellMatrix, X, U, &openmpResult);
            int successGPU = Vector_equals(Y, Z);
            int successOpenMP = Vector_equals(Z, U);
            fprintf(out, "{\n");
            fprintf(out, "\"matrix\": \"%s\",\n", entry->d_name);
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
            fprintf(out, "\"GPUCSRresult\": ");
            SpMVResultCUDA_outAsJSON(&gpuCsrResult, out);
            fprintf(out, ",\n");
            fprintf(out, "\"OpenMPresult\": ");
            SpMVResultCPU_outAsJSON(&openmpResult, out);
            fprintf(out, "\n},\n");
            ELLMatrix_free(ellMatrix);
            CSRMatrix_free(csrMatrix);
            Vector_pinned_memory_free(X);
            Vector_pinned_memory_free(Y);
            Vector_free(Z);
            Vector_free(U);
            COOMatrix_free(cooMatrix);
            MTXParser_free(mtxParser);
            fprintf(out, "{}\n]}\n");
            print_status_bar(numDir, numDir, "");
            fprintf(stderr, "\n");
        }
    }
    return EXIT_SUCCESS;
}
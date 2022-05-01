#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "Vector.h"
#include "ELLMatrix.h"
#include "SpMV.h"
#include "util.h"

#define PROGRAM_NAME "spmvELL"
#define MATRIX_SPLIT_THRESHOLD 32
#define MAX_ITERATION 512


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
    int numDir = count_file_in_directory(argv[1]) - 2;
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
        Vector *X = Vector_new_wpm(cooMatrix->col_size);
        if (!X) {
            fprintf(stderr, "Vector_new_wpm(%lu)", cooMatrix->row_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(X, 1.0f);
        Vector *Y = Vector_new_wpm(cooMatrix->row_size);
        if (!Y) {
            fprintf(stderr, "Vector_new_wpm(%lu)", cooMatrix->col_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(Y, 0.0f);
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
        SpMVResultCUDA gpuResult;
        if (noSplit) {
            ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO_wpm(cooMatrix);
            if (!ellMatrix) {
                perror("ELLMatrix_new()");
                exit(EXIT_FAILURE);
            }
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(ellMatrix);
            Vector *d_x = Vector_to_CUDA(X);
            Vector *d_y = Vector_to_CUDA(Y);
            for (int i = 0; i < MAX_ITERATION; i++) {
                ELLMatrix_SpMV_CUDA(d_ellMatrix, d_x, d_y, &gpuResult);
                if (i != MAX_ITERATION - 1) {
                    Vector_free_CUDA(d_x);
                    d_x = d_y;
                    d_y = Vector_to_CUDA(Y);
                }
            }
            Vector_free_CUDA(d_x);
            Vector_free_CUDA(d_y);
            ELLMatrix_free_wpm(ellMatrix);
        } else {
            ELLMatrix *ellLower = ELLMatrix_new_fromCOO_wpm(lower);
            if (!ellLower) {
                perror("ELLMatrix_new()");
                exit(EXIT_FAILURE);
            }
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(ellLower);
            for (int i = 0; i < MAX_ITERATION; i++) {
                ELLCOOMatrix_SpMV_CUDA(ellLower, higher, X, Y, &gpuResult);
                if (i != MAX_ITERATION - 1) {
                    Vector_free_wpm(X);
                    X = Y;
                    Y = Vector_new_wpm(cooMatrix->row_size);
                }
            }
            ELLMatrix_free_CUDA(d_ellMatrix);
            ELLMatrix_free_wpm(ellLower);
        }
        fprintf(out, "{\n");
        fprintf(out, "\"matrix\": \"%s\",\n", entry->d_name);
        fprintf(out, "\"split\": %s,\n", (noSplit) ? "true" : "false");
        fprintf(out, "\"MatrixInfo\": ");
        COOMatrix_infoOutAsJSON(cooMatrix, out);
        fprintf(out, ",\n");
        fprintf(out, "\"GPUresult\": ");
        SpMVResultCUDA_outAsJSON(&gpuResult, out);
        fprintf(out, "\n},\n");
        Vector_free_wpm(Y);
        Vector_free_wpm(X);
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
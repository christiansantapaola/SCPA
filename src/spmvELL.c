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
#include "util.h"
#include "cudaUtils.h"

#define PROGRAM_NAME "spmvELL"
#define MAX_ITERATION 512
#define ELL_THRESHOLD 256

void outAsJSON(char *absolutePath, COOMatrix *matrix, u_int64_t nz, int hybrid, double time, int numIteration, int isFirst, int isLast, FILE *out) {
    if (isFirst) {
        fprintf(out, "{ \"ELLResult\": [\n");
    }
    fprintf(out, "{\n");
    fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
    fprintf(out, "\"MatrixInfo\": ");
    COOMatrix_infoOutAsJSON(matrix, out);
    fprintf(out, ",\n");
    fprintf(out, "\"numIteration\": %d,\n", numIteration);
    fprintf(out, "\"Hybrid\": \"%s\",\n", (hybrid) ? "True" : "False");
    fprintf(out, "\"time\": %f,\n", time);
    fprintf(out, "\"meanTime\": %f,\n", time / (double)numIteration);
    fprintf(out, "\"FLOPS\": %f\n", compute_FLOPS(nz, (time / (double)numIteration) / 1000.0));
    if (!isLast) {
        fprintf(out, "},\n");
    } else {
        fprintf(out, "}\n]}\n");
    }
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
    int numDir = count_file_in_directory(argv[1]) - 2;
    struct dirent *entry;
    FILE *out = (argc >= 3) ? fopen(argv[2], "wb+") : stdout;
    if (!out) {
        perror(argv[2]);
        closedir(dir);
        return EXIT_FAILURE;
    }
    char absolutePath [PATH_MAX+1];
    chdir(argv[1]);
    int fileProcessed = 0;
    while ((entry = readdir(dir)) != NULL) {
        // if file is not a regular file then skip.
        if (entry->d_type != DT_REG) {
            continue;
        }
        // update status bar
        print_status_bar(fileProcessed, numDir, entry->d_name);
        fileProcessed++;
        // get the fullpath of the current file.
        memset(absolutePath, 0, PATH_MAX + 1);
        char *ptr = realpath(entry->d_name, absolutePath);
        if (!ptr) {
            perror(entry->d_name);
            exit(EXIT_FAILURE);
        }
        COOMatrix *h_cooMatrix = read_matrix_from_file(ptr);
        Vector *h_x = Vector_new_wpm(h_cooMatrix->row_size);
        if (!h_x) {
            fprintf(stderr, "Vector_new_wpm(%lu)", h_cooMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_x, 1.0f);
        Vector *h_y = Vector_new_wpm(h_cooMatrix->col_size);
        if (!h_y) {
            fprintf(stderr, "Vector_new_wpm(%lu)", h_cooMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_y, 0.0f);

        COOMatrix *h_low, *h_high;
        h_low = COOMatrix_new();
        h_high = COOMatrix_new();
        int notSplit = COOMatrix_split(h_cooMatrix, h_low, h_high, ELL_THRESHOLD);
        if (notSplit == -1) {
            return EXIT_FAILURE;
        }
        double totTime = 0.0;
        if (notSplit) {
            ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO(h_cooMatrix);
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
            ELLMatrix_free(h_ellMatrix);
            int cudaDev = CudaUtils_getBestDevice(d_ellMatrix->data_size * sizeof(float) + (h_x->size + h_y->size) * sizeof(float));
            CudaUtils_setDevice(cudaDev);
            for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
                float time;
                ELLMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, h_x, h_y, &time);
                totTime += time;
            }

            ELLMatrix_free_CUDA(d_ellMatrix);
        } else {
            ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO(h_low);
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
            ELLMatrix_free(h_ellMatrix);
            int cudaDev = CudaUtils_getBestDevice(d_ellMatrix->data_size * sizeof(float) + (h_x->size + h_y->size) * sizeof(float));
            CudaUtils_setDevice(cudaDev);
            for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
                float time;
                ELLCOOMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, h_high, h_x, h_y, &time);
                totTime += time;
            }
            ELLMatrix_free_CUDA(d_ellMatrix);
        }
        COOMatrix_free(h_low);
        COOMatrix_free(h_high);

        int hybrid = !notSplit;
        outAsJSON(absolutePath, h_cooMatrix,h_cooMatrix->num_non_zero_elements, hybrid,  totTime, MAX_ITERATION, fileProcessed == 1, fileProcessed == numDir, out);

        COOMatrix_free(h_cooMatrix);
        Vector_free_wpm(h_x);
        Vector_free_wpm(h_y);
    }
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
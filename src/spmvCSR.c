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

#define PROGRAM_NAME "spmvCSR"
#define MAX_ITERATION 512

void outAsJSON(char *absolutePath, CSRMatrix *matrix, float time, int numIteration, int isFirst, int isLast, FILE *out) {
    if (isFirst) {
        fprintf(out, "{ \"CSRResult\": [\n");
    }
    fprintf(out, "{\n");
    fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
    fprintf(out, "\"MatrixInfo\": ");
    CSRMatrix_infoOutAsJSON(matrix, out);
    fprintf(out, ",\n");
    fprintf(out, "\"time\": %f,\n", time);
    fprintf(out, "\"meanTime\": %f\n", time / (float)numIteration);
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
        CSRMatrix *h_csrMatrix = read_csrMatrix_from_file(ptr);
        Vector *h_x = Vector_new_wpm(h_csrMatrix->row_size);
        if (!h_x) {
            fprintf(stderr, "Vector_new_wpm(%lu)", h_csrMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_x, 1.0f);
        Vector *h_y = Vector_new_wpm(h_csrMatrix->col_size);
        if (!h_y) {
            fprintf(stderr, "Vector_new_wpm(%lu)", h_csrMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_y, 0.0f);
        int cudaDev = CudaUtils_getBestDevice(h_csrMatrix->num_non_zero_elements * sizeof(float) + (h_x->size + h_y->size) * sizeof(float));
        CudaUtils_setDevice(cudaDev);
        CSRMatrix *d_csrMatrix = CSRMatrix_to_CUDA(h_csrMatrix);
        CSRMatrix_free(h_csrMatrix);
        float totTime = 0.0f;
        for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
            float time;
            CSRMatrix_SpMV_CUDA(cudaDev, d_csrMatrix, h_x, h_y, &time);
            totTime += time;
        }
        outAsJSON(absolutePath, d_csrMatrix, totTime, MAX_ITERATION, fileProcessed == 1, fileProcessed == numDir, out);
        CSRMatrix_free_CUDA(d_csrMatrix);
        Vector_free_wpm(h_x);
        Vector_free_wpm(h_y);
    }
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
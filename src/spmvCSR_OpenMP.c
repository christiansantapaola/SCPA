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

void outAsJSON(char *absolutePath, CSRMatrix *matrix, u_int64_t nz, float time, int numIteration, int isFirst, int isLast, FILE *out) {
    if (isFirst) {
        fprintf(out, "{ \"CSRResult\": [\n");
    }
    fprintf(out, "{\n");
    fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
    fprintf(out, "\"MatrixInfo\": ");
    CSRMatrix_infoOutAsJSON(matrix, out);
    fprintf(out, ",\n");
    fprintf(out, "\"numIteration\": %d,\n", numIteration);
    fprintf(out, "\"time\": %f,\n", time);
    fprintf(out, "\"meanTime\": %f,\n", time / (float)numIteration);
    fprintf(out, "\"FLOPS\": %f\n", compute_FLOPS(nz, (time / (float)numIteration) / 1000.0));
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
        COOMatrix *cooMatrix = read_matrix_from_file(ptr);
        CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
        Vector *x = Vector_new(csrMatrix->row_size);
        if (!x) {
            fprintf(stderr, "Vector_new(%lu)", csrMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(x, 1.0f);
        Vector *y = Vector_new(csrMatrix->col_size);
        if (!y) {
            fprintf(stderr, "Vector_new(%lu)", csrMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(y, 0.0f);
        float totTime = 0.0f;
        for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
            float time;
            CSRMatrix_SpMV(csrMatrix, x, y, &time, 1);
            totTime += time;
        }
        outAsJSON(absolutePath, csrMatrix ,csrMatrix->num_non_zero_elements, totTime, MAX_ITERATION, fileProcessed == 1, fileProcessed == numDir, out);
        CSRMatrix_free(csrMatrix);
        COOMatrix_free(cooMatrix);
        Vector_free(x);
        Vector_free(y);
    }
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
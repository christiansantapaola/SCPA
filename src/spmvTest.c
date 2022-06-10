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

#define PROGRAM_NAME "spmvTest"
#define MAX_ITERATION 512

void outAsJSON(char *absolutePath, COOMatrix *matrix, int csrSuccess, int ellSuccess, int isFirst, int isLast, FILE *out) {
    if (isFirst) {
        fprintf(out, "{ \"CSRResult\": [\n");
    }
    fprintf(out, "{\n");
    fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
    fprintf(out, "\"MatrixInfo\": ");
    COOMatrix_infoOutAsJSON(matrix, out);
    fprintf(out, ",\n");
    fprintf(out, "\"csrSuccess\": \"%s\",\n", (csrSuccess) ? "true" : "false");
    fprintf(out, "\"ellSuccess\": \"%s\"\n", (ellSuccess) ? "true" : "false");
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
        COOMatrix *matrix = read_matrix_from_file(ptr);
        Vector *h_x = Vector_new(matrix->row_size);
        if (!h_x) {
            fprintf(stderr, "Vector_new_wpm(%lu)", matrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_x, 1.0f);
        Vector *h_y = Vector_new(matrix->col_size);
        if (!h_y) {
            fprintf(stderr, "Vector_new(%lu)", matrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_y, 0.0f);
        Vector *h_y_csr = Vector_new_wpm(matrix->col_size);
        if (!h_y_csr) {
            fprintf(stderr, "Vector_new_wpm(%lu)", matrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_y_csr, 0.0f);

        Vector *h_y_ell = Vector_new_wpm(matrix->col_size);
        if (!h_y_ell) {
            fprintf(stderr, "Vector_new_wpm(%lu)", matrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_y_ell, 0.0f);

        COOMatrix_SpMV(matrix, h_x, h_y, NULL);

        CSRMatrix *h_csr = CSRMatrix_new(matrix);
        CSRMatrix *d_csr = CSRMatrix_to_CUDA(h_csr);        
        CSRMatrix_free(h_csr);
        CSRMatrix_SpMV_CUDA(0, d_csr, h_x, h_y_csr, NULL);
        CSRMatrix_free_CUDA(d_csr);

        ELLCOOMatrix_SpMV(matrix, h_x, h_y_ell, 512, 1);
        int success_csr = Vector_equals(h_y, h_y_csr);
        int success_ell = Vector_equals(h_y, h_y_ell);

        outAsJSON(absolutePath, matrix, success_csr, success_ell, fileProcessed == 1, fileProcessed == numDir, out);
        COOMatrix_free(matrix);
        Vector_free(h_x);
        Vector_free(h_y);
        Vector_free_wpm(h_y_csr);
        Vector_free_wpm(h_y_ell);

    }
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
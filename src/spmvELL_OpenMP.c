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

#include <omp.h>

#define PROGRAM_NAME "spmvELL"
#define MAX_ITERATION 512
#define ELL_THRESHOLD 256

void outAsJSON(char *absolutePath, COOMatrix *matrix, u_int64_t nz, double time, int numIteration, int isFirst, int isLast, FILE *out) {
    if (isFirst) {
        fprintf(out, "{ \"ELLResult\": [\n");
    }
    fprintf(out, "{\n");
    fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
    fprintf(out, "\"MatrixInfo\": ");
    COOMatrix_infoOutAsJSON(matrix, out);
    fprintf(out, ",\n");
    fprintf(out, "\"numIteration\": %d,\n", numIteration);
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
        COOMatrix *cooMatrix = read_matrix_from_file(ptr);
        Vector *x = Vector_new(cooMatrix->row_size);
        if (!x) {
            fprintf(stderr, "Vector_new(%lu)", cooMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(x, 1.0f);
        Vector *y = Vector_new(cooMatrix->col_size);
        if (!y) {
            fprintf(stderr, "Vector_new(%lu)", cooMatrix->row_size);
            perror(" ");
            exit(EXIT_FAILURE);
        }
        Vector_set(y, 0.0f);
        COOMatrix *low = COOMatrix_new();
        COOMatrix *high = COOMatrix_new();
        int notSplit = COOMatrix_split(cooMatrix, low, high, ELL_THRESHOLD);
        if (notSplit == -1) {
            return EXIT_FAILURE;
        }
        double totTime = 0.0;
        if (notSplit) {
            ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(cooMatrix);
            for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
                float time;
                ELLMatrix_SpMV(ellMatrix, x, y, &time, 1);
                totTime += time;
            }

            ELLMatrix_free(ellMatrix);
        } else {
            ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(low);
            Vector *tmp = Vector_new(y->size);
            for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
                Vector_copy(tmp, y);
                double start = omp_get_wtime();
                #pragma omp parallel default(shared)
                #pragma omp single 
                {
                    #pragma omp task 
                    {
                    ELLMatrix_SpMV(ellMatrix, x, y, NULL, 1);
                    }
                    #pragma omp task 
                    {
                    COOMatrix_SpMV(high, x, tmp, NULL);
                    }
                    #pragma taskwait
                    Vector_sum(y, tmp);
                }
                double end = omp_get_wtime();
                totTime += (end - start) * 1000.0;
            }
            Vector_free(tmp);
            ELLMatrix_free(ellMatrix);
        }
    
        COOMatrix_free(low);
        COOMatrix_free(high);
        outAsJSON(absolutePath, cooMatrix, cooMatrix->num_non_zero_elements, totTime, MAX_ITERATION, fileProcessed == 1, fileProcessed == numDir, out);
        Vector_free(x);
        Vector_free(y);
        COOMatrix_free(cooMatrix);
    }
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
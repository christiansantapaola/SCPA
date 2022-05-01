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

#define PROGRAM_NAME "spmvCSR"
#define MAX_ITERATION 512

void outAsJSON(char *absolutePath, CSRMatrix *csrMatrix, SpMVResultCUDA *gpuResult,int isFirst, int isLast, FILE *out) {
    if (isFirst) {
        fprintf(out, "{ \"CSRResult\": [\n");
    }
    fprintf(out, "{\n");
    fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
    fprintf(out, "\"MatrixInfo\": ");
    CSRMatrix_infoOutAsJSON(csrMatrix, out);
    fprintf(out, ",\n");
    fprintf(out, "\"GPUresult\": ");
    SpMVResultCUDA_outAsJSON(gpuResult, out);
    if (!isLast) {
        fprintf(out, "\n},\n");
    } else {
        fprintf(out, "n}\n]}\n");
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
        Vector *h_x = Vector_new_wpm(cooMatrix->row_size);
        if (!h_x) {
            fprintf(stderr, "Vector_new_wpm(%lu)", cooMatrix->row_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_x, 1.0f);
        Vector *h_y = Vector_new_wpm(cooMatrix->col_size);
        if (!h_y) {
            fprintf(stderr, "Vector_new_wpm(%lu)", cooMatrix->col_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(h_y, 0.0f);
        SpMVResultCUDA gpuResult= {0};
        Vector *d_x = Vector_to_CUDA(h_x);
        Vector *d_y = Vector_to_CUDA(h_y);
        CSRMatrix *h_csrMatrix = CSRMatrix_new_wpm(cooMatrix);
        if (!h_csrMatrix) {
            perror("CSRMatrix_new_wpm()");
            exit(EXIT_FAILURE);

        }
        CSRMatrix *d_csrMatrix = CSRMatrix_to_CUDA(h_csrMatrix);
        Vector *zeroes = Vector_new_wpm(h_csrMatrix->row_size);
        Vector_set(zeroes, 0.0f);
        for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
            SpMVResultCUDA gpuResultTmp;
            CSRMatrix_SpMV_CUDA(d_csrMatrix, d_x, d_y, &gpuResultTmp);
            gpuResult.GPUKernelExecutionTime += gpuResultTmp.GPUKernelExecutionTime;
        }
        Vector *z = Vector_from_CUDA(d_y);
        int isFirst = fileProcessed == 0;
        int isLast = fileProcessed == numDir;
        outAsJSON(absolutePath, h_csrMatrix, &gpuResult, isFirst, isLast, out);
        CSRMatrix_free_CUDA(d_csrMatrix);
        CSRMatrix_free_wpm(h_csrMatrix);
        Vector_free_wpm(h_x);
        Vector_free_wpm(h_y);
        Vector_free(z);
        COOMatrix_free(cooMatrix);
        MTXParser_free(mtxParser);
    }
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
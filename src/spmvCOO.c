#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "Vector.h"
#include "SpMV.h"
#include "util.h"

#define PROGRAM_NAME "spmvCOO"
#define MAX_ITERATION 512

void outAsJSON(char *absolutePath, COOMatrix *cooMatrix, SpMVResultCPU *cpuResult,int isFirst, int isLast, FILE *out) {
    if (isFirst) {
        fprintf(out, "{ \"CSRResult\": [\n");
    }
    fprintf(out, "{\n");
    fprintf(out, "\"matrix\": \"%s\",\n", absolutePath);
    fprintf(out, "\"MatrixInfo\": ");
    COOMatrix_infoOutAsJSON(cooMatrix, out);
    fprintf(out, ",\n");
    fprintf(out, "\"CPUresult\": ");
    SpMVResultCPU_outAsJSON(cpuResult, out);
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
        Vector *x = Vector_new(cooMatrix->row_size);
        if (!x) {
            fprintf(stderr, "Vector_new(%lu)", cooMatrix->row_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(x, 1.0f);
        Vector *y = Vector_new(cooMatrix->col_size);
        if (!y) {
            fprintf(stderr, "Vector_new(%lu)", cooMatrix->col_size);
            perror("");
            exit(EXIT_FAILURE);
        }
        Vector_set(y, 0.0f);
        SpMVResultCPU cpuResult = {0};
        Vector_set(y, 1.0f);
        cpuResult.timeElapsed = 0;
        for (u_int64_t i = 0; i < MAX_ITERATION; i++) {
            SpMVResultCPU cpuResultTmp;
            COOMatrix_SpMV(cooMatrix, x, y, &cpuResultTmp);
            cpuResult.timeElapsed += cpuResultTmp.timeElapsed;
            if (i != MAX_ITERATION - 1) {
                Vector_free(x);
                x = y;
                y = Vector_new(cooMatrix->row_size);
            }
        }
        int isFirst = fileProcessed == 0;
        int isLast = fileProcessed == numDir;
        outAsJSON(absolutePath, cooMatrix, &cpuResult, isFirst, isLast, out);
        Vector_free(x);
        Vector_free(y);
        COOMatrix_free(cooMatrix);
        MTXParser_free(mtxParser);
    }
    print_status_bar(numDir, numDir, "");
    fprintf(stderr, "\n");
    return EXIT_SUCCESS;
}
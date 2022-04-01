//
// Created by 9669c on 01/04/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include "COOMatrix.h"

const char *PROGRAM_NAME = "readCOO";

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
    FILE *mtx = fopen(argv[1], "r");
    if (!mtx) {
        perror("fopen() failed");
        return EXIT_FAILURE;
    }
    COOMatrix *cooMatrix = COOMatrix_new(mtx);
    if (!cooMatrix) {
        perror("newCOOMatrix() failed");
        fclose(mtx);
        return EXIT_FAILURE;
    }
    if (mtx != stdin) {
        fclose(mtx);
    }
    COOMatrix_outAsJSON(cooMatrix, stdout);
    putc('\n', stdout);
    COOMatrix_free(cooMatrix);
    return EXIT_SUCCESS;
}

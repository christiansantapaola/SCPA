//
// Created by 9669c on 26/03/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include "COOMatrix.h"
#include "CSRMatrix.h"

const char *PROGRAM_NAME = "readCSR";

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("fopen()");
        return 1;
    }
    COOMatrix *cooMatrix = COOMatrix_new(file);
    if (file != stdin) {
        fclose(file);
    }
    CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
    CSRMatrix_outAsJSON(csrMatrix, stdout);
    putc('\n', stdout);
    COOMatrix_free(cooMatrix);
    CSRMatrix_free(csrMatrix);
    return EXIT_SUCCESS;
}

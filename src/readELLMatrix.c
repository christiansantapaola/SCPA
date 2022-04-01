//
// Created by 9669c on 26/03/2022.
//
//
// Created by 9669c on 26/03/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"


const char *PROGRAM_NAME = "readELL";
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("fopen()");
        return EXIT_FAILURE;
    }
    COOMatrix *cooMatrix = COOMatrix_new(file);
    if (file != stdin) {
        fclose(file);
    }
    CSRMatrix *csrMatrix = CSRMatrix_new(cooMatrix);
    ELLMatrix *ellMatrix = ELLMatrix_new(csrMatrix);
    ELLMatrix_outAsJSON(ellMatrix, stdout);
    putc('\n', stdout);
    COOMatrix_free(cooMatrix);
    CSRMatrix_free(csrMatrix);
    ELLMatrix_free(ellMatrix);

    return EXIT_SUCCESS;
}


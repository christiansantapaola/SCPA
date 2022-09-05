//
// Created by 9669c on 26/03/2022.
//
//
// Created by 9669c on 26/03/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "util.h"

#define PROGRAM_NAME "readELL"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
    COOMatrix *cooMatrix = read_matrix_from_file(argv[1]);
    if (!cooMatrix) {
        perror("read_matrix_from_file");
        return EXIT_FAILURE;
    }
    ELLMatrix *ellMatrix = ELLMatrix_new_fromCOO(cooMatrix);
    ELLMatrix_outAsJSON(ellMatrix, stdout);
    putc('\n', stdout);
    COOMatrix_free(cooMatrix);
    ELLMatrix_free(ellMatrix);
    return EXIT_SUCCESS;
}


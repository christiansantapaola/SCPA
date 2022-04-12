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

#define PROGRAM_NAME "readELL"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "USAGE: %s file.mtx\n", PROGRAM_NAME);
        return EXIT_FAILURE;
    }
    MTXParser *mtxParser = MTXParser_new(argv[1]);
    if (!mtxParser) {
        perror("MTXParser_new()");
        exit(EXIT_FAILURE);
    }
    COOMatrix *cooMatrix = MTXParser_parse(mtxParser);
    if (!cooMatrix) {
        perror("MTXParser_parse():");
        MTXParser_free(mtxParser);
        return EXIT_FAILURE;
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


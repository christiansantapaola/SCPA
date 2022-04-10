//
// Created by 9669c on 26/03/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "MTXParser.h"
const char *PROGRAM_NAME = "readCSR";

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
    CSRMatrix_outAsJSON(csrMatrix, stdout);
    putc('\n', stdout);
    COOMatrix_free(cooMatrix);
    CSRMatrix_free(csrMatrix);
    return EXIT_SUCCESS;
}

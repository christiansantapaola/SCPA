//
// Created by 9669c on 01/04/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include "MTXParser.h"
#include "COOMatrix.h"

#define PROGRAM_NAME "readCOO"

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
    COOMatrix_outAsJSON(cooMatrix, stdout);
    putc('\n', stdout);
    COOMatrix_free(cooMatrix);
    MTXParser_free(mtxParser);
    return EXIT_SUCCESS;
}

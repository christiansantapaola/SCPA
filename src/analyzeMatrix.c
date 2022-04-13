#include <stdio.h>
#include <stdlib.h>
#include "MTXParser.h"
#include "COOMatrix.h"
#include "Histogram.h"


#define PROGRAM_NAME "analyzeMatrix"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "%s file.mtx\n", PROGRAM_NAME);
        exit(EXIT_FAILURE);
    }
    MTXParser *parser = MTXParser_new(argv[1]);
    if (!parser) {
        perror(argv[1]);
        exit(EXIT_FAILURE);
    }
    COOMatrix *matrix = MTXParser_parse(parser);
    if (!matrix) {
        exit(EXIT_FAILURE);
    }
    COOMatrix_outAsJSON(matrix, stdout);
    COOMatrix first, second;
    int ret = COOMatrix_split(matrix, &first, &second, 5);
    if (ret == -1) {
        fprintf(stderr, "fail\n");
        return EXIT_FAILURE;
    }
    COOMatrix_outAsJSON(&first, stdout);
    COOMatrix_outAsJSON(&second, stdout);


    return 0;
}
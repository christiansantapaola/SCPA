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

    Histogram *rowForElem = Histogram_new(matrix->col_size);
    for (u_int64_t elem = 0; elem < matrix->num_non_zero_elements; elem++) {
        Histogram_insert(rowForElem, matrix->row_index[elem]);
    }
    Histogram_outAsJSON(rowForElem, stdout);
    return 0;
}
#ifndef SPARSEMATRIX_MTXPARSER_H
#define SPARSEMATRIX_MTXPARSER_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "COOMatrix.h"

#define MTXPARSER_MAX_LINE_SIZE 4096

typedef struct MTXParser {
    FILE *file;
    char *filename;
    int currentLine;
    char *line;
    size_t lineSize;
} MTXParser;

MTXParser *MTXParser_new(char *filename);
void MTXParser_free(MTXParser *mtxParser);
COOMatrix *MTXParser_parse(MTXParser *mtxParser);
int MTXParser_parseLine(MTXParser *mtxParser, u_int64_t *row, u_int64_t *col, float * data);

#endif //SPARSEMATRIX_MTXPARSER_H

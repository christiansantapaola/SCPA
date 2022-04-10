//
// Created by 9669c on 07/04/2022.
//

#ifndef SPARSEMATRIX_MTXPARSER_H
#define SPARSEMATRIX_MTXPARSER_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "COOMatrix.h"

typedef struct MTXParser {
    FILE *file;
    char *filename;
    int currentLine;
    char *line;
    char *invalidToken;
} MTXParser;

MTXParser *MTXParser_new(char *filename);
void MTXParser_free(MTXParser *mtxParser);
COOMatrix *MTXParser_parse(MTXParser *mtxParser);
int MTXParser_parseLine(MTXParser *parserStatus, u_int64_t *row, u_int64_t *col, float * data);

#endif //SPARSEMATRIX_MTXPARSER_H

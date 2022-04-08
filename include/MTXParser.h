//
// Created by 9669c on 07/04/2022.
//

#ifndef SPARSEMATRIX_MTXPARSER_H
#define SPARSEMATRIX_MTXPARSER_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int MTXParseLine(FILE *f, u_int64_t *row, u_int64_t *col, float * data);

#endif //SPARSEMATRIX_MTXPARSER_H

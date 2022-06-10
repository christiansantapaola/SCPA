//
// Created by 9669c on 20/04/2022.
//

#ifndef SPARSEMATRIX_UTIL_H
#define SPARSEMATRIX_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#include "MTXParser.h"
#include "COOMatrix.h"
#include "CSRMatrix.h"

void print_status_bar(int used, int total,char *file);
int count_file_in_directory(const char *dirpath);
COOMatrix *read_matrix_from_file(const char *path);
CSRMatrix *read_csrMatrix_from_file(const char *path);
double compute_FLOPS(u_int64_t nz, float time);
double compute_mean(float *obs, unsigned int size);
double compute_var(float *obs, unsigned int size, double mean);
#endif //SPARSEMATRIX_UTIL_H

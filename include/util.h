//
// Created by 9669c on 20/04/2022.
//

#ifndef SPARSEMATRIX_UTIL_H
#define SPARSEMATRIX_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>

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
int forEachFile(const char *path, int (*op)(char *, struct dirent *, void *), void *args );
void transposef(float *dest, const float *src, u_int64_t num_row, u_int64_t num_col);
void transpose_u_int64_t(u_int64_t *dest, const u_int64_t *src, u_int64_t num_row, u_int64_t num_col);

#endif //SPARSEMATRIX_UTIL_H

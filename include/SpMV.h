#ifndef SPARSEMATRIX_SPMV_H
#define SPARSEMATRIX_SPMV_H

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"
#include "cudaUtils.h"

#define SPMV_FAIL -1
#define SPMV_SUCCESS 0

int COOMatrix_SpMV(const COOMatrix *matrix, const Vector *x, Vector *y, float *time);

int CSRMatrix_SpMV(const CSRMatrix *matrix, const Vector *x, Vector *y, float *time, int parallel);

int ELLMatrix_SpMV(const ELLMatrix *matrix, const Vector *x, Vector *y, float *time, int parallel);

int ELLCOOMatrix_SpMV(COOMatrix *h_cooMatrix, Vector *h_x, Vector *h_y, u_int64_t threshold, u_int64_t max_iteration);


int CSRMatrix_SpMV_CUDA(int cudaDevice, const CSRMatrix *d_matrix, const Vector *h_x, Vector *h_y, float *time);
int ELLMatrix_SpMV_CUDA(int cudaDevice, const ELLMatrix *d_matrix, const Vector *h_x, Vector *h_y, float *time);
int ELLCOOMatrix_SpMV_CUDA(int cudaDevice, const ELLMatrix *d_ellMatrix, const COOMatrix *h_cooMatrix, const Vector *h_x, Vector *h_y, float *time);

#endif //SPARSEMATRIX_SPMV_H

#ifndef SPARSEMATRIX_SPMV_H
#define SPARSEMATRIX_SPMV_H

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"

#define SPMV_FAIL -1
#define SPMV_SUCCESS 0

int COOMatrix_SpMV(const COOMatrix *matrix, const Vector *x, Vector *y);

int CSRMatrix_SpMV(const CSRMatrix *matrix, const Vector *x, Vector *y);

int ELLMatrix_SpMV(const ELLMatrix *matrix, const Vector *x, Vector *y);

int CSRMatrix_SpMV_CUDA(int cudaDevice, const CSRMatrix *d_matrix, const Vector *h_x, Vector *h_y);
int ELLMatrix_SpMV_CUDA(int cudaDevice, const ELLMatrix *d_matrix, const Vector *h_x, Vector *h_y);
int ELLCOOMatrix_SpMV_CUDA(int cudaDevice, const ELLMatrix *d_ellMatrix, const COOMatrix *h_cooMatrix, const Vector *h_x, Vector *h_y);

#endif //SPARSEMATRIX_SPMV_H

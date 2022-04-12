#ifndef SPARSEMATRIX_SPMV_H
#define SPARSEMATRIX_SPMV_H

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"

void CSRMatrix_SpMV_CPU(const CSRMatrix *matrix, const Vector *x, Vector *y, SpMVResultCPU *result);
void CSRMatrix_SpMV_OPENMP(const CSRMatrix *matrix, const Vector *x, Vector *y, SpMVResultCPU *result);
void CSRMatrix_SpMV_GPU(const CSRMatrix *matrix, const Vector *x, Vector *y, SpMVResultCUDA *result);

void ELLMatrix_SpMV_CPU(const ELLMatrix *matrix, const Vector *x, Vector *y, SpMVResultCPU *result);
void ELLMatrix_SpMV_OPENMP(const ELLMatrix *matrix, const Vector *x, Vector *y, SpMVResultCPU *result);
void ELLMatrix_SpMV_GPU(const ELLMatrix *matrix, const Vector *x, Vector *y, SpMVResultCUDA *result);

#endif //SPARSEMATRIX_SPMV_H

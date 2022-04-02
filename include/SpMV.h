//
// Created by 9669c on 01/04/2022.
//

#ifndef SPARSEMATRIX_SPMV_H
#define SPARSEMATRIX_SPMV_H

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"
#include "SpMVResult.h"

void CSRMatrix_SpMV_CPU(const CSRMatrix *matrix,const Vector *x, Vector *y, SpMVResult *result);
void CSRMatrix_SpMV_CPU_OPENMP(const CSRMatrix *matrix,const Vector *x, Vector *y, SpMVResult *result);
void CSRMatrix_SpMV_GPU(const CSRMatrix *matrix,const Vector *x, Vector *y, SpMVResult *result);

void ELLMatrix_SpMV_CPU(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResult *result);
void ELLMatrix_SpMV_CPU_OPENMP(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResult *result);
void ELLMatrix_SpMV_GPU(const ELLMatrix *matrix,const Vector *x, Vector *y, SpMVResult *result);

#endif //SPARSEMATRIX_SPMV_H

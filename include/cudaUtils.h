//
// Created by 9669c on 12/05/2022.
//

#ifndef SPARSEMATRIX_CUDAUTILS_H
#define SPARSEMATRIX_CUDAUTILS_H
#include <stdlib.h>

void CudaUtils_setDevice(int device);
int CudaUtils_getBestDevice(size_t memoryUsed);

#endif //SPARSEMATRIX_CUDAUTILS_H

//
// Created by 9669c on 16/03/2022.
//

#ifndef SPARSEMATRIX_CUDAUTILS_H
#define SPARSEMATRIX_CUDAUTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
extern "C" {
#include <stdlib.h>
#include "BlockGridInfo.h"
};

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void CudaUtils_setDevice(int device);
void CudaUtils_getDeviceProp(int device, cudaDeviceProp *prop);
int CudaUtils_getBestDevice();
void CudaUtils_getBestCudaParameters(unsigned int numRows, cudaDeviceProp *prop, BlockGridInfo *bestParams);










#endif //SPARSEMATRIX_CUDAUTILS_H

#ifndef SPARSEMATRIX_CUDAUTILS_CUH
#define SPARSEMATRIX_CUDAUTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
extern "C" {
#include <stdlib.h>
#include "BlockGridInfo.h"
#include "cudaUtils.h"
};

#define checkCudaErrors(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void CudaUtils_getDeviceProp(int device, cudaDeviceProp *prop);
void CudaUtils_getBestCudaParameters(u_int64_t numUnits, cudaDeviceProp *prop, BlockGridInfo *bestParams);

#endif //SPARSEMATRIX_CUDAUTILS_CUH

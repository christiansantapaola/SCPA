//
// Created by 9669c on 16/03/2022.
//

#ifndef SPARSEMATRIX_CUDAUTILS_H
#define SPARSEMATRIX_CUDAUTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct CudaDeviceInfo {
    int numDevice;
    cudaDeviceProp *devices;
    int dev;
    int driverVersion;
    int runtimeVersion;

    CudaDeviceInfo();
    ~CudaDeviceInfo();
    void setDevice(int dev);
    cudaDeviceProp *getDeviceProp(int dev);
    int getBestDevice();
    bool doesItFitInGlobalMemory(size_t size);
    bool doesItFitInSharedMemory(size_t size);
    bool doesItFitInCostantMemory(size_t size);
    struct BlockGridInfo getBlockSize(int NumRows);
};

struct BlockGridInfo {
    size_t blockSize;
    size_t gridSize;
    float utilization;
    int wastedThread;
    unsigned long numBlockToFillSM;
    int maxThreadPerBlock;
    int maxThreadPerSM;
    size_t maxBlockSizePerSM;
    int MaxBlockPerSM;
};







#endif //SPARSEMATRIX_CUDAUTILS_H

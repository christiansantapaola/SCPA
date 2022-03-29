//
// Created by 9669c on 16/03/2022.
//

#include "cudaUtils.cuh"


CudaDeviceInfo::CudaDeviceInfo() {
    checkCudaErrors(cudaGetDeviceCount(&numDevice));
    devices = new cudaDeviceProp[numDevice];
    for (int i = 0; i < numDevice; i++) {
        checkCudaErrors(cudaGetDeviceProperties(&devices[i], i));
    }
    dev = 0;
    checkCudaErrors(cudaDriverGetVersion(&driverVersion));
    checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
}

CudaDeviceInfo::~CudaDeviceInfo() {
    delete[] devices;
}

void CudaDeviceInfo::setDevice(int dev) {
    if (dev < numDevice) {
        dev = dev;
        checkCudaErrors(cudaSetDevice(dev));
    }
}

cudaDeviceProp *CudaDeviceInfo::getDeviceProp(int dev) {
    if (dev < numDevice) {
        return &devices[dev];
    }
    return nullptr;
}

int CudaDeviceInfo::getBestDevice() {
    int bestDev = 0;
    int numSM = 0;
    int clockRate = 0;
    for (int i = 0; i < numDevice; i++) {
        if (numSM < devices[i].multiProcessorCount) {
            numSM = devices[i].multiProcessorCount;
            bestDev = i;
            clockRate = devices[i].clockRate;
        } else if ( numSM == devices[i].multiProcessorCount ) {
            if (clockRate < devices[i].clockRate) {
                numSM = devices[i].multiProcessorCount;
                bestDev = i;
                clockRate = devices[i].clockRate;
            }
        }
    }
    return bestDev;
}

bool CudaDeviceInfo::doesItFitInSharedMemory(size_t size) {
    return devices[dev].sharedMemPerBlock >= size;
}

bool CudaDeviceInfo::doesItFitInGlobalMemory(size_t size) {
    return devices[dev].totalGlobalMem >= size;
}

bool CudaDeviceInfo::doesItFitInCostantMemory(size_t size) {
    return devices[dev].totalConstMem >= size;
}


struct BlockGridInfo CudaDeviceInfo::getBlockSize(int NumRows) {
    int size = 0;
    for (size = 1;  devices[dev].warpSize * size < devices[dev].maxThreadsPerBlock; size++);
    struct BlockGridInfo *infos = new BlockGridInfo[size];
    for (int i = 1; devices[dev].warpSize * i < devices[dev].maxThreadsPerBlock; i++) {
        infos[i - 1].maxThreadPerBlock = devices[dev].maxThreadsPerBlock;
        infos[i - 1].maxBlockSizePerSM = devices[dev].maxBlocksPerMultiProcessor;
        infos[i - 1].maxThreadPerSM = devices[dev].maxThreadsPerMultiProcessor;
        infos[i - 1].blockSize = devices[dev].warpSize * i;
        infos[i - 1].numBlockToFillSM = devices[dev].maxThreadsPerMultiProcessor / infos[i - 1].blockSize;
        infos[i - 1].gridSize = (NumRows % infos[i - 1].blockSize == 0) ? NumRows / infos[i - 1].blockSize : NumRows / infos[i - 1].blockSize + 1;
        float spread = (infos[i - 1].gridSize < devices[dev].multiProcessorCount) ? (float) infos[i - 1].gridSize / (float) devices[dev].multiProcessorCount : 1.0f;
        float utilizationSM = (float) infos[i - 1].gridSize / (float) infos[i - 1].numBlockToFillSM;
        unsigned int numThread = infos[i - 1].blockSize * infos[i - 1].gridSize;
        float wastedThreadOverTotalThread = (float) (numThread - NumRows) / (float) numThread;
        infos[i - 1].utilization = utilizationSM + spread -  wastedThreadOverTotalThread;
        // printf("%s blocksize=%zu, gridsize=%zu, utilizationSM=%f, spread=%f, wastedThread=%f, utilization=%f\n", "CUDA:", infos[i - 1].blockSize, infos[i-1].gridSize, utilizationSM, spread, wastedThreadOverTotalThread, infos[i - 1].utilization);

    }
    int index = 0;
    float maxUtil = 0.0f;
    for (int i = 0; i < size; i++) {
        if (infos[i].utilization > maxUtil) {
            maxUtil = infos[i].utilization;
            index = i;
        }
    }
    BlockGridInfo gridInfo = infos[index];
    delete[] infos;
    printf("%s blocksize=%zu, gridsize=%zu\n", "CUDA:", gridInfo.blockSize, gridInfo.gridSize);
    return gridInfo;
}



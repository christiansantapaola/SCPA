//
// Created by 9669c on 16/03/2022.
//

#include "cudaUtils.cuh"


CudaDeviceInfo::CudaDeviceInfo() {
    numDevice = 0;
    driverVersion = 0;
    runtimeVersion = 0;
    dev = 0;
    checkCudaErrors(cudaGetDeviceCount(&numDevice));
    devices = new cudaDeviceProp[numDevice];
    for (int i = 0; i < numDevice; i++) {
        checkCudaErrors(cudaGetDeviceProperties(&devices[i], i));
    }
    checkCudaErrors(cudaDriverGetVersion(&driverVersion));
    checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
}

CudaDeviceInfo::~CudaDeviceInfo() {
    delete[] devices;
}

void CudaDeviceInfo::setDevice(int device) {
    if (device < numDevice) {
        this->dev = device;
        checkCudaErrors(cudaSetDevice(device));
    }
}

cudaDeviceProp *CudaDeviceInfo::getDeviceProp(int device) const {
    if (device < numDevice) {
        return &devices[device];
    }
    return nullptr;
}

int CudaDeviceInfo::getBestDevice() const {
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

bool CudaDeviceInfo::doesItFitInSharedMemory(size_t size) const {
    return devices[dev].sharedMemPerBlock >= size;
}

bool CudaDeviceInfo::doesItFitInGlobalMemory(size_t size) const {
    return devices[dev].totalGlobalMem >= size;
}

bool CudaDeviceInfo::doesItFitInCostantMemory(size_t size) const {
    return devices[dev].totalConstMem >= size;
}


struct BlockGridInfo CudaDeviceInfo::getBlockSize(int NumRows) const {
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
        infos[i - 1].spread = (infos[i - 1].gridSize < devices[dev].multiProcessorCount) ? (float) infos[i - 1].gridSize / (float) devices[dev].multiProcessorCount : 1.0f;
        infos[i - 1].utilizationSM = (float) infos[i - 1].gridSize / (float) infos[i - 1].numBlockToFillSM;
        infos[i - 1].numThread = infos[i - 1].blockSize * infos[i - 1].gridSize;
        infos[i - 1].wastedThread = infos[i - 1].numThread - NumRows;
        infos[i - 1].wastedThreadOverNumThread = (float) (infos[i - 1].wastedThread) / (float) infos[i - 1].numThread;
        infos[i - 1].utilization = infos[i - 1].utilizationSM + infos[i - 1].spread - infos[i - 1].wastedThreadOverNumThread;
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
    return gridInfo;
}


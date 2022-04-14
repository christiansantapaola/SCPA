#include "cudaUtils.cuh"


void CudaUtils_setDevice(int device) {
    checkCudaErrors(cudaSetDevice(device));
}
void CudaUtils_getDeviceProp(int device, cudaDeviceProp *prop) {
    if (!prop) return;
    checkCudaErrors(cudaGetDeviceProperties(prop, device));

}
int CudaUtils_getBestDevice(size_t memoryUsed) {
    int numDevices;
    cudaDeviceProp *props;
    checkCudaErrors(cudaGetDeviceCount(&numDevices));
    props = (cudaDeviceProp *) malloc(sizeof(*props) * numDevices);
    for (int i = 0; i < numDevices; i++) {
        CudaUtils_getDeviceProp(i, &props[i]);
    }
    int bestDev = -1;
    int numSM = 0;
    int clockRate = 0;
    for (int i = 0; i < numDevices; i++) {
        if (memoryUsed > props[i].totalGlobalMem) {
            continue;
        }
        if (numSM < props[i].multiProcessorCount) {
            numSM = props[i].multiProcessorCount;
            bestDev = i;
            clockRate = props[i].clockRate;
        } else if ( numSM == props[i].multiProcessorCount ) {
            if (clockRate < props[i].clockRate) {
                numSM = props[i].multiProcessorCount;
                bestDev = i;
                clockRate = props[i].clockRate;
            }
        }
    }
    free(props);
    return bestDev;
}

int doesItFitInGlobalMemory(cudaDeviceProp *prop, size_t size) {
    if (!prop) return 0;
    return size <= prop->totalGlobalMem;
}

void CudaUtils_getBestCudaParameters(u_int64_t numUnits, cudaDeviceProp *prop, BlockGridInfo *bestParams) {
    if (!bestParams || !prop) return;
    u_int64_t size = 0;
    for (size = 1; prop->warpSize * size <= prop->maxThreadsPerBlock; size++);
    BlockGridInfo *infos = (BlockGridInfo *)malloc(size * sizeof(BlockGridInfo));
    memset(infos, 0, size * sizeof(BlockGridInfo));
    for (u_int64_t i = 1; prop->warpSize * i <= prop->maxThreadsPerBlock; i++) {
        infos[i - 1].maxThreadPerBlock = prop->maxThreadsPerBlock;
        infos[i - 1].maxBlockSizePerSM = prop->maxBlocksPerMultiProcessor;
        infos[i - 1].maxThreadPerSM = prop->maxThreadsPerMultiProcessor;
        infos[i - 1].blockSize =prop->warpSize * i;
        infos[i - 1].numBlockToFillSM = prop->maxThreadsPerMultiProcessor / infos[i - 1].blockSize;
        infos[i - 1].gridSize = (numUnits % infos[i - 1].blockSize == 0) ? numUnits / infos[i - 1].blockSize : numUnits / infos[i - 1].blockSize + 1;
        infos[i - 1].spread = (infos[i - 1].gridSize < (u_int64_t)prop->multiProcessorCount) ? (double) infos[i - 1].gridSize / (double) prop->multiProcessorCount : 1.0;
        infos[i - 1].utilizationSM = (double) infos[i - 1].gridSize / (double) infos[i - 1].numBlockToFillSM;
        infos[i - 1].numThread = infos[i - 1].blockSize * infos[i - 1].gridSize;
        infos[i - 1].wastedThread = infos[i - 1].numThread - numUnits;
        infos[i - 1].wastedThreadOverNumThread = (double) (infos[i - 1].wastedThread) / (double) infos[i - 1].numThread;
        infos[i - 1].utilization = infos[i - 1].utilizationSM + infos[i - 1].spread - infos[i - 1].wastedThreadOverNumThread;
    }
    int index = 0;
    double maxUtil = 0.0;
    for (int i = 0; i < size; i++) {
        if (infos[i].utilization > maxUtil) {
            maxUtil = infos[i].utilization;
            index = i;
        }
    }

    *bestParams = infos[index];
    free(infos);
}
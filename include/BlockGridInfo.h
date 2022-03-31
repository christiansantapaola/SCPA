//
// Created by 9669c on 31/03/2022.
//

#ifndef SPARSEMATRIX_BLOCKGRIDINFO_H
#define SPARSEMATRIX_BLOCKGRIDINFO_H

struct BlockGridInfo {
    size_t blockSize;
    size_t gridSize;
    float utilizationSM;
    float spread;
    unsigned int numThread;
    unsigned int wastedThread;
    float wastedThreadOverNumThread;
    float utilization;
    unsigned long numBlockToFillSM;
    unsigned long maxThreadPerBlock;
    unsigned long maxBlockSizePerSM;
    unsigned long maxThreadPerSM;
};

#endif //SPARSEMATRIX_BLOCKGRIDINFO_H

#ifndef SPARSEMATRIX_BLOCKGRIDINFO_H
#define SPARSEMATRIX_BLOCKGRIDINFO_H

#include <stdio.h>
#include <stdlib.h>

typedef struct BlockGridInfo {
    u_int64_t blockSize;
    u_int64_t gridSize;
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
} BlockGridInfo;

void BlockGridInfo_outAsJSON(BlockGridInfo *blockGridInfo, FILE *out);

#endif //SPARSEMATRIX_BLOCKGRIDINFO_H

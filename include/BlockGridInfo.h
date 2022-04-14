#ifndef SPARSEMATRIX_BLOCKGRIDINFO_H
#define SPARSEMATRIX_BLOCKGRIDINFO_H

#include <stdio.h>
#include <stdlib.h>

typedef struct BlockGridInfo {
    u_int64_t blockSize;
    u_int64_t gridSize;
    double utilizationSM;
    double spread;
    u_int64_t numThread;
    u_int64_t wastedThread;
    double wastedThreadOverNumThread;
    double utilization;
    u_int64_t numBlockToFillSM;
    u_int64_t maxThreadPerBlock;
    u_int64_t maxBlockSizePerSM;
    u_int64_t maxThreadPerSM;
} BlockGridInfo;

void BlockGridInfo_outAsJSON(BlockGridInfo *blockGridInfo, FILE *out);

#endif //SPARSEMATRIX_BLOCKGRIDINFO_H

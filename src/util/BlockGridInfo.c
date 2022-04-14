#include "BlockGridInfo.h"

void BlockGridInfo_outAsJSON(BlockGridInfo *blockGridInfo, FILE *out) {
    if (!blockGridInfo || !out) return;
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %zu,\n", "\"blockSize\"",  blockGridInfo->blockSize);
    fprintf(out, "%s: %zu,\n", "\"gridSize\"",  blockGridInfo->gridSize);
    fprintf(out, "%s: %f,\n", "\"utilizationSM\"",  blockGridInfo->utilizationSM);
    fprintf(out, "%s: %f,\n", "\"spread\"",  blockGridInfo->spread);
    fprintf(out, "%s: %lu,\n", "\"numThread\"",  blockGridInfo->numThread);
    fprintf(out, "%s: %lu,\n", "\"wastedThread\"",  blockGridInfo->wastedThread);
    fprintf(out, "%s: %f,\n", "\"wastedThreadOverNumThread\"",  blockGridInfo->wastedThreadOverNumThread);
    fprintf(out, "%s: %f,\n", "\"utilization\"",  blockGridInfo->utilization);
    fprintf(out, "%s: %zu,\n", "\"numBlockToFillSM\"",  blockGridInfo->numBlockToFillSM);
    fprintf(out, "%s: %zu,\n", "\"maxThreadPerSM\"",  blockGridInfo->maxThreadPerSM);
    fprintf(out, "%s: %zu,\n", "\"maxBlockSizePerSM\"",  blockGridInfo->maxBlockSizePerSM);
    fprintf(out, "%s: %zu\n", "\"maxThreadPerBlock\"",  blockGridInfo->maxThreadPerBlock);
    fprintf(out, "%s", "}");
}

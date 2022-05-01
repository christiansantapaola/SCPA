#ifndef SPARSEMATRIX_SPMVRESULT_H
#define SPARSEMATRIX_SPMVRESULT_H

#include <stdio.h>
#include <stdlib.h>
#include "BlockGridInfo.h"

typedef struct SpMVResultCUDA {
    float GPUKernelExecutionTime;
    float TotalTime;
    float CPUTime;
} SpMVResultCUDA;

typedef struct SpMVResultCPU {
    float timeElapsed;
} SpMVResultCPU;

void SpMVResultCUDA_outAsJSON(SpMVResultCUDA *result, FILE *out);
void SpMVResultCPU_outAsJSON(SpMVResultCPU *result, FILE *out);



#endif //SPARSEMATRIX_SPMVRESULT_H

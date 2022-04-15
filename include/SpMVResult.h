#ifndef SPARSEMATRIX_SPMVRESULT_H
#define SPARSEMATRIX_SPMVRESULT_H

#include <stdio.h>
#include <stdlib.h>
#include "BlockGridInfo.h"

typedef struct SpMVResultCUDA {
    int success;
    float GPUInputOnDeviceTime;
    float GPUKernelExecutionTime;
    float GPUOutputFromDeviceTime;
    float GPUTotalTime;
} SpMVResultCUDA;

typedef struct SpMVResultOpenMP {
    int success;
    float timeElapsed;
    int numThreads;
} SpMVResultOpenMP;

typedef struct SpMVResultCPU {
    int success;
    float timeElapsed;
} SpMVResultCPU;

void SpMVResultCUDA_outAsJSON(SpMVResultCUDA *result, FILE *out);
void SpMVResultOpenMP_outAsJSON(SpMVResultOpenMP *result, FILE *out);
void SpMVResultCPU_outAsJSON(SpMVResultCPU *result, FILE *out);



#endif //SPARSEMATRIX_SPMVRESULT_H

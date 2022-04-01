//
// Created by 9669c on 23/03/2022.
//

#ifndef SPARSEMATRIX_SPMVRESULT_H
#define SPARSEMATRIX_SPMVRESULT_H

#include <stdio.h>
#include <stdlib.h>
#include "BlockGridInfo.h"

typedef struct SpMVResult {
    int success;
    float GPUInputOnDeviceTime;
    float GPUKernelExecutionTime;
    float GPUOutputFromDeviceTime;
    BlockGridInfo blockGridInfo;
    size_t GPUtotalGlobMemory;
    size_t GPUusedGlobalMemory;
    float CPUFunctionExecutionTime;
} SpMVResult;

void SpMVResult_outAsJSON(SpMVResult *result, FILE *out);


#endif //SPARSEMATRIX_SPMVRESULT_H

//
// Created by 9669c on 23/03/2022.
//

#ifndef SPARSEMATRIX_SPMVRESULT_H
#define SPARSEMATRIX_SPMVRESULT_H
#include <ostream>
#include <inttypes.h>


struct SpMVResult {
    bool success;
    float GPUInputOnDeviceTime;
    float GPUKernelExecutionTime;
    float GPUOutputFromDeviceTime;
    int gridSize;
    int blockSize;
    int wastedThread;
    size_t GPUtotalGlobMemory;
    size_t GPUusedGlobalMemory;
    float CPUFunctionExecutionTime;


    friend std::ostream& operator<<(std::ostream& out, SpMVResult& result);
};


#endif //SPARSEMATRIX_SPMVRESULT_H

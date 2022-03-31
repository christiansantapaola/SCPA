//
// Created by 9669c on 23/03/2022.
//

#include "SpMVResult.h"

std::ostream& operator<<(std::ostream& out, SpMVResult& result) {
    out << "{" << std::endl;
    out << "\"success\": " << ((result.success) ? "true" : "false") << "," <<std::endl;
    out << "\"CPUFunctionExecutionTime\": " << result.CPUFunctionExecutionTime << "," <<std::endl;
    out << "\"GPUInputOnDeviceTime\": " << result.GPUInputOnDeviceTime << "," <<std::endl;
    out << "\"GPUKernelExecutionTime\": " << result.GPUKernelExecutionTime << "," << std::endl;
    out << "\"GPUOutputFromDeviceTime\": " << result.GPUOutputFromDeviceTime << "," << std::endl;
    out << "\"GPUTotalGlobMemory\": " << result.GPUtotalGlobMemory << "," << std::endl;
    out << "\"GPUusedGlobMemory\": " << result.GPUusedGlobalMemory << "," << std::endl;
    out << "\"GPUusedGlobMemoryRatio\": " << ((result.GPUtotalGlobMemory != 0) ? ((double) result.GPUusedGlobalMemory / (double)result.GPUtotalGlobMemory) : 0.0) << "," << std::endl;
    out << "\"blockGridInfo\": {" << std:: endl;
    out << "\"blockSize\": " << result.blockGridInfo.blockSize << "," << std::endl;
    out << "\"gridSize\": " << result.blockGridInfo.gridSize << "," << std::endl;
    out << "\"utilizationSM\": " << result.blockGridInfo.utilizationSM << "," << std::endl;
    out << "\"spread\": " << result.blockGridInfo.spread << "," << std::endl;
    out << "\"numThread\": " << result.blockGridInfo.numThread << "," << std::endl;
    out << "\"wastedThread\": " << result.blockGridInfo.wastedThread << "," << std::endl;
    out << "\"wastedThreadOverNumThread\": " << result.blockGridInfo.wastedThreadOverNumThread << "," << std::endl;
    out << "\"utilization\": " << result.blockGridInfo.utilization << "," << std::endl;
    out << "\"numBlockToFillSM\": " << result.blockGridInfo.numBlockToFillSM << "," << std::endl;
    out << "\"maxThreadPerSM\": " << result.blockGridInfo.maxThreadPerSM << "," << std::endl;
    out << "\"maxBlockSizePerSM\": " << result.blockGridInfo.maxBlockSizePerSM << "," << std::endl;
    out << "\"maxThreadPerBlock\": " << result.blockGridInfo.maxThreadPerBlock << std::endl;
    out << "}" << std::endl;
    // out << "\"\": " << "," << std::endl;
    out << "}";
    return out;
}

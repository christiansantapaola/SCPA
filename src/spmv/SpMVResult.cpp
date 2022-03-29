//
// Created by 9669c on 23/03/2022.
//

#include "SpMVResult.h"

std::ostream& operator<<(std::ostream& out, SpMVResult& result) {
    out << "{" << std::endl;
    out << "\"success\": " << ((result.success) ? "true" : "false") << "," <<std::endl;
    out << "\"CPUFunctionExecutionTime\": " << result.CPUFunctionExecutionTime << "," <<std::endl;
    out << "\"GPUInputOnDeviceTime\": " << result.GPUInputOnDeviceTime << "," <<std::endl;
    out << "\"GPUKernelExecutionTime\": " << result.GPUKernelExecutionTime << "," <<std::endl;
    out << "\"GPUOutputFromDeviceTime\": " << result.GPUOutputFromDeviceTime <<std::endl;
    out << "}";

    return out;
}

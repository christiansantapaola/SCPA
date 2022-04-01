//
// Created by 9669c on 23/03/2022.
//

#include "SpMVResult.h"

void SpMVResult_outAsJSON(SpMVResult *result, FILE *out) {
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %s,\n", "\"success\"",  ((result->success) ? "true" : "false"));
    fprintf(out, "%s: %f,\n", "\"CPUFunctionExecutionTime\"",  result->CPUFunctionExecutionTime);
    fprintf(out, "%s: %f,\n", "\"GPUInputOnDeviceTime\"",  result->GPUInputOnDeviceTime);
    fprintf(out, "%s: %f,\n", "\"GPUKernelExecutionTime\"",  result->GPUKernelExecutionTime);
    fprintf(out, "%s: %f,\n", "\"GPUOutputFromDeviceTime\"",  result->GPUOutputFromDeviceTime);
    fprintf(out, "%s: %zu,\n", "\"GPUtotalGlobMemory\"",  result->GPUtotalGlobMemory);
    fprintf(out, "%s: %zu,\n", "\"GPUusedGlobalMemory\"",  result->GPUusedGlobalMemory);
    fprintf(out, "%s: %f,\n", "\"GPUusedGlobMemoryRatio\"",  ((result->GPUtotalGlobMemory != 0) ? ((double) result->GPUusedGlobalMemory / (double)result->GPUtotalGlobMemory) : 0.0));
    fprintf(out, "%s: ", "\"BlockGridInfo\"");
    BlockGridInfo_outAsJSON(&result->blockGridInfo, out);
    fprintf(out, "\n");
    fprintf(out, "%s", "}");
}

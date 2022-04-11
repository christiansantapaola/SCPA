//
// Created by 9669c on 23/03/2022.
//

#include "SpMVResult.h"

void SpMVResultCUDA_outAsJSON(SpMVResultCUDA *result, FILE *out) {
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %s,\n", "\"success\"",  ((result->success) ? "true" : "false"));
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

void SpMVResultOpenMP_outAsJSON(SpMVResultOpenMP *result, FILE *out) {
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %s,\n", "\"success\"",  ((result->success) ? "true" : "false"));
    fprintf(out, "%s: %f,\n", "\"timeElapsed\"", result->timeElapsed);
    fprintf(out, "%s: %d\n", "\"numThreads\"",  result->numThreads);
    fprintf(out, "\n");
    fprintf(out, "%s", "}");
}

void SpMVResultCPU_outAsJSON(SpMVResultCPU *result, FILE *out) {
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %s,\n", "\"success\"",  ((result->success) ? "true" : "false"));
    fprintf(out, "%s: %f\n", "\"timeElapsed\"", result->timeElapsed);
    fprintf(out, "\n");
    fprintf(out, "%s", "}");
}

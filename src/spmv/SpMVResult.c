#include "SpMVResult.h"

void SpMVResultCUDA_outAsJSON(SpMVResultCUDA *result, FILE *out) {
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %s,\n", "\"success\"",  ((result->success) ? "true" : "false"));
    fprintf(out, "%s: %f,\n", "\"GPUInputOnDeviceTime\"",  result->GPUInputOnDeviceTime);
    fprintf(out, "%s: %f,\n", "\"GPUKernelExecutionTime\"",  result->GPUKernelExecutionTime);
    fprintf(out, "%s: %f,\n", "\"GPUOutputFromDeviceTime\"",  result->GPUOutputFromDeviceTime);
    fprintf(out, "%s: %f\n", "\"GPUTotalTime\"",  result->GPUTotalTime);
    fprintf(out, "%s: %f\n", "\"CPUTime\"",  result->CPUTime);
    fprintf(out, "%s", "}");
}

void SpMVResultOpenMP_outAsJSON(SpMVResultOpenMP *result, FILE *out) {
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %s,\n", "\"success\"",  ((result->success) ? "true" : "false"));
    fprintf(out, "%s: %f,\n", "\"timeElapsed\"", result->timeElapsed);
    fprintf(out, "%s: %d\n", "\"numThreads\"",  result->numThreads);
    fprintf(out, "\n");
    fprintf(out, "%s", "}");
}

void SpMVResultCPU_outAsJSON(SpMVResultCPU *result, FILE *out) {
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %s,\n", "\"success\"",  ((result->success) ? "true" : "false"));
    fprintf(out, "%s: %f\n", "\"timeElapsed\"", result->timeElapsed);
    fprintf(out, "%s", "}");
}

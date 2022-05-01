#include "SpMVResult.h"

void SpMVResultCUDA_outAsJSON(SpMVResultCUDA *result, FILE *out) {
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %f,\n", "\"GPUKernelExecutionTime\"",  result->GPUKernelExecutionTime);
    fprintf(out, "%s: %f,\n", "\"CPUTime\"",  result->CPUTime);
    fprintf(out, "%s: %f\n", "\"TotalTime\"",  result->TotalTime);
    fprintf(out, "%s", "}");
}

void SpMVResultCPU_outAsJSON(SpMVResultCPU *result, FILE *out) {
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %f\n", "\"timeElapsed\"", result->timeElapsed);
    fprintf(out, "%s", "}");
}

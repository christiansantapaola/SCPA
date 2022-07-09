#include "SpMV.h"

int ELLCOOMatrix_SpMV(COOMatrix *h_cooMatrix, Vector *h_x, Vector *h_y, u_int64_t threshold, u_int64_t max_iteration) {
        COOMatrix *h_low, *h_high;
        h_low = COOMatrix_new();
        h_high = COOMatrix_new();
        int notSplit = COOMatrix_split(h_cooMatrix, h_low, h_high, threshold);
        if (notSplit == -1) {
            return EXIT_FAILURE;
        }
        double totTime = 0.0;
        if (notSplit) {
            ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO(h_cooMatrix);
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
            int cudaDev = CudaUtils_getBestDevice(d_ellMatrix->data_size * sizeof(float) + (h_x->size + h_y->size) * sizeof(float));
            CudaUtils_setDevice(cudaDev);
            for (u_int64_t i = 0; i < max_iteration; i++) {
                float time;
                ELLMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, h_x, h_y, &time);
                totTime += time;
            }
            ELLMatrix_free_CUDA(d_ellMatrix);
            ELLMatrix_free(h_ellMatrix);
        } else {
            ELLMatrix *h_ellMatrix = ELLMatrix_new_fromCOO(h_low);
            ELLMatrix *d_ellMatrix = ELLMatrix_to_CUDA(h_ellMatrix);
            int cudaDev = CudaUtils_getBestDevice(d_ellMatrix->data_size * sizeof(float) + (h_x->size + h_y->size) * sizeof(float));
            CudaUtils_setDevice(cudaDev);
            for (u_int64_t i = 0; i < max_iteration; i++) {
                float time;
                ELLCOOMatrix_SpMV_CUDA(cudaDev, d_ellMatrix, h_high, h_x, h_y, &time);
                totTime += time;
            }
            ELLMatrix_free_CUDA(d_ellMatrix);
            ELLMatrix_free(h_ellMatrix);
        }
        COOMatrix_free(h_low);
        COOMatrix_free(h_high);
        return SPMV_SUCCESS;
}
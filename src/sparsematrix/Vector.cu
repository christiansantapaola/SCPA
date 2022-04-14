extern "C" {
#include "Vector.h"
}
#include <cuda.h>
#include "cudaUtils.cuh"

Vector *Vector_new_wpm(unsigned int size) {
    Vector *vector = (Vector *) malloc(sizeof(Vector));
    vector->size = size;
    checkCudaErrors(cudaHostAlloc(&vector->data, size * sizeof(float), cudaHostAllocDefault));
    memset(vector->data, 0, size * sizeof(float));
    return vector;
}
void Vector_free_wpm(Vector *vector) {
    if (!vector) return;
    checkCudaErrors(cudaFreeHost(vector->data));
    free(vector);
}

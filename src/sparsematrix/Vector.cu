extern "C" {
#include "Vector.h"
}
#include <cuda.h>
#include "cudaUtils.cuh"

Vector *Vector_new_wpm(u_int64_t size) {
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

Vector *Vector_to_CUDA(const Vector *h_vector) {
    Vector *d_vector = (Vector *) malloc(sizeof(Vector));
    d_vector->size = h_vector->size;
    checkCudaErrors(cudaMalloc(&d_vector->data, h_vector->size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_vector->data, h_vector->data, d_vector->size * sizeof(float), cudaMemcpyHostToDevice));
    return d_vector;
}

Vector *Vector_from_CUDA(const Vector *d_vector) {
    if (!d_vector) return NULL;
    Vector *h_vector = (Vector *) malloc(sizeof(Vector));
    if (!h_vector) return NULL;
    h_vector->size = d_vector->size;
    h_vector->data = (float *)malloc(h_vector->size * sizeof(float));
    if (!h_vector->data) {
        free(h_vector);
        return NULL;
    }
    checkCudaErrors(cudaMemcpy(h_vector->data, d_vector->data, d_vector->size * sizeof(float), cudaMemcpyDeviceToHost));
    return h_vector;
}

void Vector_copy_from_CUDA(Vector *h_vector, const Vector *d_vector) {
    if (!h_vector || !d_vector) return;
    if (h_vector == d_vector) return;
    if (h_vector->size != d_vector->size) return;
    checkCudaErrors(cudaMemcpy(h_vector->data, d_vector->data, d_vector->size * sizeof(float), cudaMemcpyDeviceToHost));
}

void Vector_free_CUDA(Vector *d_vector) {
    if (!d_vector) return;
    checkCudaErrors(cudaFree(d_vector->data));
    free(d_vector);
}

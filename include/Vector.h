#ifndef SPARSEMATRIX_VECTOR_H
#define SPARSEMATRIX_VECTOR_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>


typedef struct Vector {
    float *data;
    u_int64_t size;
} Vector;

Vector *Vector_new(u_int64_t size);
void Vector_free(Vector *vector);
Vector *Vector_pinned_memory_new(unsigned int size);
void Vector_pinned_memory_free(Vector *vector);
    void Vector_set(Vector *vector, float value);
    void Vector_outAsJSON(Vector *vector, FILE *out);
    int Vector_equals(const Vector *v1, const Vector *v2);
//    void swap(SwapMap& swapMap);
//    void swapInverse(SwapMap& swapMap);


#endif //SPARSEMATRIX_VECTOR_H

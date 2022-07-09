#ifndef SPARSEMATRIX_VECTOR_H
#define SPARSEMATRIX_VECTOR_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>

#define VECTOR_PRECISION 0.00001

typedef struct Vector {
    float *data;
    u_int64_t size;
} Vector;

Vector *Vector_new(u_int64_t size);
void Vector_free(Vector *vector);
Vector *Vector_new_wpm(u_int64_t size);
void Vector_free_wpm(Vector *vector);

Vector *Vector_to_CUDA(const Vector *h_vector);
Vector *Vector_to_CUDA_async(const Vector *h_vector);
Vector *Vector_from_CUDA(const Vector *d_vector);
void Vector_copy_from_CUDA(Vector *h_vector, const Vector *d_vector);
void Vector_free_CUDA(Vector *d_vector);

void Vector_set(Vector *vector, float value);
void Vector_outAsJSON(Vector *vector, FILE *out);
int Vector_equals(const Vector *v1, const Vector *v2);
int Vector_sum(Vector *v1, Vector *v2);
int Vector_copy(Vector *dst, const Vector *src);

#endif
//SPARSEMATRIX_VECTOR_H

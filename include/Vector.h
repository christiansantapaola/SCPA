//
// Created by 9669c on 15/03/2022.
//

#ifndef SPARSEMATRIX_VECTOR_H
#define SPARSEMATRIX_VECTOR_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>


typedef struct Vector {
    float *data;
    unsigned int size;
} Vector;

Vector *Vector_new(unsigned int size);
    void Vector_free();
    void Vector_set(Vector *vector, float value);
    void Vector_outAsJSON(Vector *vector, FILE *out);
    int Vector_equals(const Vector *v1, const Vector *v2);
//    void swap(SwapMap& swapMap);
//    void swapInverse(SwapMap& swapMap);


#endif //SPARSEMATRIX_VECTOR_H

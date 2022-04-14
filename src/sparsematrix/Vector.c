#include "Vector.h"

Vector *Vector_new(u_int64_t size) {
    Vector *vector = malloc(sizeof(Vector));
    vector->size = size;
    vector->data = malloc(size * sizeof(float));
    if (!vector->data) {
        return NULL;
    }
    memset(vector->data, 0, size * sizeof(float));
    return vector;
}

void Vector_free(Vector *vector) {
    if (!vector) return;
    free(vector->data);
    free(vector);
}

int Vector_equals(const Vector *v1, const Vector *v2) {
    if (v1 == v2) return 1;
    if (!v1 || !v2) return 0;
    int areSameSize = v1->size == v2->size;
    int areSameBuffer = v1->data == v2->data;
    if (!areSameSize) return 0;
    if (areSameBuffer) return 1;
    for (u_int64_t i = 0; i < v1->size; i++) {
        if (fabsf(v1->data[i] - v2->data[i]) > VECTOR_PRECISION) {
            return 0;
        }
    }
    return 1;
}


void Vector_set(Vector *vector, float value) {
    for (u_int64_t i = 0; i < vector->size; i++) {
        vector->data[i] = value;
    }
}

void Vector_outAsJSON(Vector *vector, FILE *out) {
    if (!out) out = stdout;
    if (!vector) {
        fprintf(out, "{}");
    }
    fprintf(out, "%s\n", "{");
    fprintf(out, "%s: %lu,\n", "\"size\"",  vector->size);
    fprintf(out, "%s: [ ", "\"data\"");
    for (u_int64_t i = 0; i < vector->size - 1; i++) {
        fprintf(out, "%f, ", vector->data[i]);
    }
    fprintf(out, "%f ],\n", vector->data[vector->size - 1]);
    fprintf(out, "%s", "}");
}
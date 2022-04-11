//
// Created by 9669c on 18/03/2022.
//

#ifndef SPARSEMATRIX_HISTOGRAM_H
#define SPARSEMATRIX_HISTOGRAM_H

#include "stdio.h"
#include "stdlib.h"

struct Pair {
    u_int64_t first;
    u_int64_t second;
};

typedef struct Histogram {
    struct Pair *hist;
    u_int64_t size;
} Histogram;

    Histogram *Histogram_new(u_int64_t size);
    void Histogram_free(Histogram *histogram);
    void Histogram_insert(Histogram *histogram, u_int64_t i);
    u_int64_t Histogram_getElemAtIndex(Histogram *histogram, u_int64_t i);
    void Histogram_outAsJSON(Histogram *histogram, FILE *out);

#endif //SPARSEMATRIX_HISTOGRAM_H

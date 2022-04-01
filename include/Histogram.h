//
// Created by 9669c on 18/03/2022.
//

#ifndef SPARSEMATRIX_HISTOGRAM_H
#define SPARSEMATRIX_HISTOGRAM_H

#include "stdio.h"
#include "stdlib.h"

struct Pair {
    int first;
    int second;
};

typedef struct Histogram {
    struct Pair *hist;
    unsigned int size;
} Histogram;

    Histogram *Histogram_new(unsigned int size);
    void Histogram_free(Histogram *histogram);
    void Histogram_insert(Histogram *histogram, int i);
    int Histogram_getElemAtIndex(Histogram *histogram, int i);
    void Histogram_outAsJSON(Histogram *histogram, FILE *out);

#endif //SPARSEMATRIX_HISTOGRAM_H

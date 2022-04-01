//
// Created by 9669c on 18/03/2022.
//

#include "Histogram.h"

Histogram *Histogram_new(unsigned int size) {
    Histogram *histogram = malloc(sizeof(Histogram));
    if (!histogram) {
        return NULL;
    }
    histogram->size = size;
    histogram->hist = malloc(size * sizeof(struct Pair));
    for (int i = 0; i < size; i++) {
        histogram->hist[i].first = 0;
        histogram->hist[i].second = i;
    }
    return histogram;
}

void Histogram_free(Histogram *histogram) {
    if (!histogram) return;
    free(histogram->hist);
    free(histogram);
}

void Histogram_insert(Histogram *histogram, int i) {
    if (!histogram) return;
    if (i < histogram->size) {
        histogram->hist[i].first++;
    }
}

int Histogram_getElemAtIndex(Histogram *histogram, int i) {
    if (i < histogram->size) {
        return histogram->hist[i].first;
    } else {
        return -1;
    }
}

void Histogram_outAsJSON(Histogram *histogram, FILE *out) {
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %u,\n", "\"size\"",  histogram->size);
    fprintf(out, "%s: [ ", "\"hist\"");
    for (int i=0; i < histogram->size - 1; i++) {
        fprintf(out, "{%d, %d}, ", histogram->hist[i].first, histogram->hist[i].second);
    }
    fprintf(out, "{%d, %d} ],\n", histogram->hist[histogram->size - 1].first, histogram->hist[histogram->size - 1].second);
    fprintf(out, "%s", "}");
}
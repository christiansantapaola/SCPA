#include "Histogram.h"

Histogram *Histogram_new(u_int64_t size) {
    Histogram *histogram = malloc(sizeof(Histogram));
    if (!histogram) {
        return NULL;
    }
    histogram->size = size;
    histogram->hist = malloc(size * sizeof(struct Pair));
    for (u_int64_t i = 0; i < size; i++) {
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

void Histogram_insert(Histogram *histogram, u_int64_t i) {
    if (!histogram) return;
    if (i < histogram->size) {
        histogram->hist[i].first++;
    }
}

u_int64_t Histogram_getElemAtIndex(Histogram *histogram, u_int64_t i) {
    if (i > histogram->size) {
        return 0;
    }
    return histogram->hist[i].first;
}

void Histogram_outAsJSON(Histogram *histogram, FILE *out) {
    fprintf(out, "%s\n", "{ ");
    fprintf(out, "%s: %lu,\n", "\"size\"",  histogram->size);
    fprintf(out, "%s: [ ", "\"hist\"");
    for (u_int64_t i=0; i < histogram->size - 1; i++) {
        fprintf(out, "{%lu, %lu}, ", histogram->hist[i].first, histogram->hist[i].second);
    }
    fprintf(out, "{%lu, %lu} ],\n", histogram->hist[histogram->size - 1].first, histogram->hist[histogram->size - 1].second);
    fprintf(out, "%s", "}");
}
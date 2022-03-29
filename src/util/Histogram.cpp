//
// Created by 9669c on 18/03/2022.
//

#include "Histogram.h"

Histogram::Histogram(unsigned int size) {
    this->size = size;
    hist = new std::pair<int, int>[size];
    for (int i = 0; i < size; i++) {
        hist[i].first = 0;
        hist[i].second = i;
    }
}

Histogram::~Histogram() {
    delete[] hist;
}

void Histogram::insert(int i) {
    if (i < size) {
        hist[i].first++;
    }
    // std::sort(&hist[0], &hist[size - 1]);
}

int Histogram::getElemAtIndex(int i) {
    return hist[i].first;
}

std::ostream& operator<<(std::ostream& out, Histogram &histogram) {
    out << "{" << std::endl;
    out << "size = " << histogram.size << std::endl;
    out << "hist = [";
    for (int i = 0; i < histogram.size - 1; i++) {
        out << "(" << histogram.hist[i].first << ", " << histogram.hist[i].second << "), ";
    }
    out << "(" << histogram.hist[histogram.size - 1].first << ", " << histogram.hist[histogram.size - 1].second << ")" << "]" << std::endl;
    out << "}" << std::endl;
    return out;
}
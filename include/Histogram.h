//
// Created by 9669c on 18/03/2022.
//

#ifndef SPARSEMATRIX_HISTOGRAM_H
#define SPARSEMATRIX_HISTOGRAM_H

#include <iostream>
#include <algorithm>

class Histogram {
    std::pair<int, int> *hist;
    unsigned int size;
public:
    Histogram(unsigned int size);
    ~Histogram();
    void insert(int i);
    int getElemAtIndex(int i);
    friend std::ostream& operator<<(std::ostream& out, Histogram &histogram);
};


#endif //SPARSEMATRIX_HISTOGRAM_H

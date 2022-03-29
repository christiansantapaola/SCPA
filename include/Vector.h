//
// Created by 9669c on 15/03/2022.
//

#ifndef SPARSEMATRIX_VECTOR_H
#define SPARSEMATRIX_VECTOR_H
#include <iostream>

#include <cmath>

#include "SwapMap.h"

class Vector {
private:
    float *data;
    unsigned int size;

public:
    explicit Vector(unsigned int size);
    ~Vector();
    void set(float value);
    unsigned int getSize();
    float *getData();
    friend std::ostream& operator<< (std::ostream &out, Vector const& matrix);
    bool equals(const Vector &v);
    void swap(SwapMap& swapMap);
    void swapInverse(SwapMap& swapMap);
};


#endif //SPARSEMATRIX_VECTOR_H
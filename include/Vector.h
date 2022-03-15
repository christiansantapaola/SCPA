//
// Created by 9669c on 15/03/2022.
//

#ifndef SPARSEMATRIX_VECTOR_H
#define SPARSEMATRIX_VECTOR_H
#include <iostream>

#include <math.h>

class Vector {
private:
    float *data;
    unsigned int size;

public:
    Vector(unsigned int size);
    ~Vector();
    void set(float value);
    unsigned int getSize();
    float *getData();
    void print();
    bool equals(const Vector &v);
};


#endif //SPARSEMATRIX_VECTOR_H

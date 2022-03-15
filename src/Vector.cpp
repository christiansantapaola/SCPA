//
// Created by 9669c on 15/03/2022.
//

#include "Vector.h"

Vector::Vector(unsigned int size) {
    this->size = size;
    data = new float[size];
}

Vector::~Vector() {
    delete data;
}

bool Vector::equals(const Vector &v) {
    bool areSameSize = size == v.size;
    bool areSameBuffer = data == v.data;
    if (!areSameSize) return false;
    if (areSameBuffer) return true;
    for (int i = 0; i < size; i++) {
        if (fabs(data[i] - v.data[i]) > 0.000001) {
            return false;
        }
    }
    return true;


}


void Vector::set(float value) {
    for (unsigned int i = 0; i < size; i++) {
        data[i] = value;
    }
}

unsigned int Vector::getSize() {
    return size;
}

float *Vector::getData() {
    return data;
}

void Vector::print() {
    std::cout << "( ";
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << ")" << std::endl;
}
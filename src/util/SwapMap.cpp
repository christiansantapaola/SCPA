//
// Created by 9669c on 17/03/2022.
//

#include "SwapMap.h"

SwapMap::SwapMap(unsigned int size) {
    this->size = size;
    this->map = new int[size];
    this->inverse = new int[size];
    memset(map, 0, size);
    memset(inverse, 0, size);
}
SwapMap::~SwapMap() {
    delete[] map;
    delete[] inverse;
}
void SwapMap::setMapping(int i, int j) {
    if (i < size && j < size) {
        map[i] = j;
        inverse[j] = i;
    }
}
int SwapMap::getMapping(int i) {
    if (i < size) {
        return map[i];
    }
    return -1;
}

int SwapMap::getInverse(int i) {
    if (i < size) {
        return inverse[i];
    }
    return -1;
}
std::ostream& operator<< (std::ostream &out, SwapMap const& swapMap) {
    out << "{" << std::endl;
    out << "size = " << swapMap.size << ", " << std::endl;
    out << "map = [ ";
    for (int i = 0; i < swapMap.size - 1; i++) {
        out << swapMap.map[i] << ", ";
    }
    out << swapMap.map[swapMap.size - 1] << "], " << std::endl;
    out << "inverse = [ ";
    for (int i = 0; i < swapMap.size - 1; i++) {
        out << swapMap.inverse[i] << ", ";
    }
    out << swapMap.inverse[swapMap.size - 1] << "], " << std::endl;
    out << "}" << std::endl;
    return out;
}

unsigned int SwapMap::getSize() {
    return size;
}
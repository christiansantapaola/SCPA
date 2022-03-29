//
// Created by 9669c on 17/03/2022.
//

#ifndef SPARSEMATRIX_SWAPMAP_H
#define SPARSEMATRIX_SWAPMAP_H

#include <cstring>
#include <iostream>

class SwapMap {
    int *map;
    int *inverse;
    unsigned int size;
public:
    SwapMap() = default;
    SwapMap(unsigned int size);
    ~SwapMap();
    void setMapping(int i, int j);
    int getMapping(int i);
    int getInverse(int i);
    unsigned int getSize();
    friend std::ostream& operator<< (std::ostream &out, SwapMap const& swapMap);
};


#endif //SPARSEMATRIX_SWAPMAP_H

//
// Created by 9669c on 24/03/2022.
//

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"

float TEST_MATRIX[4][4] = {11.0f, 12.0f, 0.0f, 0.0f,
                           0.0f, 22.0f, 23.0f, 0.0f,
                           0.0f, 0.0f, 33.0f, 0.0f,
                           0.0f, 0.0f, 43.0f, 44.0f};


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "USAGE: SparseMatrixExec file.mtx" << std::endl;
        return 1;
    }
    std::cout << "START OPENING FILE : " << argv[1] << std::endl;
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("fopen()");
        return 1;
    }
    std::cout << "FILE SUCCESSFULLY OPEN" << std::endl;
    // COOMatrix cooMatrix = COOMatrix((float*)TEST_MATRIX, 4, 4);
    std::cout << "START READING FILE" << std::endl;
    COOMatrix cooMatrix = COOMatrix(file);
    if (file != stdin) {
        fclose(file);
    }
    std::cout << "FILE READ SUCCESSFULLY" << std::endl;
    CSRMatrix csrMatrix = CSRMatrix(cooMatrix);
    std::cout << "rows=" << csrMatrix.getRowSize() << std::endl;
    std::cout << "cols=" << csrMatrix.getColSize() << std::endl;
    std::cout << "nz=" << csrMatrix.getNumNonZeroElements() << std::endl;
    Vector X = Vector(cooMatrix.getRowSize());
    Vector Y = Vector(cooMatrix.getRowSize());
    Vector Z = Vector(cooMatrix.getRowSize());
    X.set(1.0);
    Y.set(0.0f);
    Z.set(0.0f);
    SpMVResult result1 = csrMatrix.SpMV_GPU(X, Y);
    SpMVResult result2 = csrMatrix.SpMV_CPU(X, Z);
    if (Y.equals(Z)) {
        std::cout << "Y and Z are Equals" << std::endl;
        std::cout << "Y = " << Y;
        std::cout << "GPU Result = "<< result1 << std::endl;
        std::cout << "CPU Result = " << result2 << std::endl;
    } else {
        std::cout << "Y = " << Y;
        std::cout << "GPU Result = "<< result1 << std::endl;
        std::cout << "Z = " << Z;
        std::cout << "CPU Result = "<< result2 << std::endl;
    }


}
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

const char *PROGRAM_NAME = "spmvCSR";

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "USAGE: " << PROGRAM_NAME << " file.mtx" << std::endl;
        return 1;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("fopen()");
        return 1;
    }
    // COOMatrix cooMatrix = COOMatrix((float*)TEST_MATRIX, 4, 4);
    COOMatrix cooMatrix = COOMatrix(file);
    if (file != stdin) {
        fclose(file);
    }
    CSRMatrix csrMatrix = CSRMatrix(cooMatrix);
    Vector X = Vector(cooMatrix.getColSize());
    Vector Y = Vector(cooMatrix.getRowSize());
    Vector Z = Vector(cooMatrix.getRowSize());
    X.set(1.0);
    Y.set(0.0f);
    Z.set(0.0f);
    SpMVResult gpuResult = csrMatrix.SpMV_GPU(X, Y);
    SpMVResult cpuResult = csrMatrix.SpMV_CPU(X, Z);
    if (Y.equals(Z)) {
        std::cout << "{" << std::endl;
        std::cout << "\"success\": true," << std::endl;
        std::cout << "\"rows\":" << csrMatrix.getRowSize() << "," << std::endl;
        std::cout << "\"cols\":" << csrMatrix.getColSize() << "," << std::endl;
        std::cout << "\"num_non_zero_elements\":" << csrMatrix.getNumNonZeroElements()  << "," << std::endl;
        std::cout << "\"GPU_Result\":" << gpuResult << ","<< std::endl;
        std::cout << "\"CPU_Result\":" <<  cpuResult << std::endl;
        std::cout << "}" << std::endl;
    } else {
        std::cout << "{" << std::endl;
        std::cout << "\"success\": false," << std::endl;
        std::cout << "\"Y\": " << Y << ","<< std::endl;
        std::cout << "\"GPU_Result\": " << gpuResult << ","<< std::endl;
        std::cout << "\"Z\":" << Z << ","<< std::endl;
        std::cout << "\"CPU_Result\":" << cpuResult << std::endl;
        std::cout << "}" << std::endl;
        exit(EXIT_FAILURE);
    }


}
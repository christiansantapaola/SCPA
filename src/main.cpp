#include <stdio.h>
#include <iostream>

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"
#include "Vector.h"

float TEST_MATRIX[4][4] = {3.0f, 0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 2.0f, 4.0f, 1.0f,
                           1.0f, 0.0f, 0.0f, 1.0f};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "USAGE: SparseMatrixExec file.mtx" << std::endl;
        return 1;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("fopen()");
        return 1;
    }
    COOMatrix cooMatrix = COOMatrix((float *)TEST_MATRIX, 4 ,4);
    if (file != stdin) {
        fclose(file);
    }
    CSRMatrix csrMatrix = CSRMatrix(cooMatrix);
    Vector X = Vector(csrMatrix.getRowSize());
    Vector Y = Vector(csrMatrix.getRowSize());
    Vector Z = Vector(csrMatrix.getRowSize());
    X.set(1.0f);
    Y.set(0.0f);
    Z.set(0.0f);
    csrMatrix.SpMV(X.getData(), Y.getData());
    ELLMatrix ellMatrix = ELLMatrix(csrMatrix);
    std::cout << "COO Matrix" << std::endl;
    std::cout << cooMatrix;
    std::cout << "CSR Matrix" << std::endl;
    std::cout << csrMatrix;
    std::cout << "ELL Matrix" << std::endl;
    std::cout << ellMatrix;
    ellMatrix.SpMV(X, Z);
    if (Y.equals(Z)) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        std::cout << "Printing result of Sequential SpMV" << std::endl;
        std::cout << Y;
        std::cout << std::endl;
        std::cout << Z;
        std::cout << "Printing result of Parallel SpMV" << std::endl;
    }

}
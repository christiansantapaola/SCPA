//
// Created by 9669c on 26/03/2022.
//
//
// Created by 9669c on 26/03/2022.
//

#include <iostream>
#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "ELLMatrix.h"

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
    COOMatrix cooMatrix = COOMatrix(file);
    if (file != stdin) {
        fclose(file);
    }
    CSRMatrix csrMatrix = CSRMatrix(cooMatrix);
    ELLMatrix ellMatrix = ELLMatrix(csrMatrix);
    std::cout << ellMatrix << std::endl;
}


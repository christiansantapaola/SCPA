#include <stdio.h>
#include <iostream>

#include "COOMatrix.h"
#include "CSRMatrix.h"
#include "Vector.h"


void vec_print(float *vec, int size) {
    if (!vec) return;
    printf("%s", "( ");
    for (int i = 0; i < size; i++) {
        printf("%.3f ", vec[i]);
    }
    printf("%s\n", ")");
}

void vec_init(float *vec, int size, float value) {
    if (!vec) return;
    for (int i = 0; i < size; i++) {
        vec[i] = value;
    }
}

//int main(int argc, char *argv[])
//{
//    struct COOMatrix matrix = {0};
//    //    struct ELLMatrix ellMatrix = {0};
//    struct CSRMatrix csrMatrix = {0};
//    float *X = NULL;
//    float *Y = NULL;
////    float X[4] = {1.0f,1.0f,1.0f,1.0f};
////    float Y[4] = {0.0f,0.0f,0.0f,0.0f};
////    float MATRIX[4][4] = {3.0f, 0.0f, 1.0f, 0.0f,
////                          0.0f, 0.0f, 0.0f, 0.0f,
////                          0.0f, 2.0f, 4.0f, 1.0f,
////                          1.0f, 0.0f, 0.0f, 1.0f};
////    float Z[4] = {0.0f,0.0f,0.0f,0.0f};
//
//    if (argc < 2) {
//        fprintf(stderr, "%s\n", "usage: cmd file.mat");
//        exit(1);
//    }
//    FILE *file = fopen(argv[1], "r");
//    if (!file) {
//        perror("fopen failed!");
//        exit(1);
//    }
//    SparseMatrixCSRRead(file, &matrix);
//
//    X = malloc(matrix.col_size * sizeof (float));
//    vec_init(X, matrix.col_size, 1.0);
//    Y =  malloc(matrix.col_size * sizeof (float));
//    vec_init(Y, matrix.col_size, 0.0);
//
////    Matrix_to_COOMatrix((float *)MATRIX, 4, 4, &matrix);
//
//// COOMatrix_print(&matrix);
//
//    COOMatrix_to_CSRMatrix(&matrix, &csrMatrix);
//    CSRMatrix_print(&csrMatrix);
//
////    CSRMatrix_to_ELLMatrix(&csrMatrix, &ellMatrix);
////
////    ELLMatrix_print(&ellMatrix);
////
////    SpMV_CSR_seq(&csrMatrix, X, Y);
////    for (int i = 0; i < 4; i++) {
////        printf("%f\n", Y[i]);
////    }
////    printf("\n");
////    SpMV_ELL_seq(&ellMatrix, X, Z);
////    for (int i = 0; i < 4; i++) {
////        printf("%f\n", Z[i]);
////    }
////   printf("\n");
//// SpMV_CSR_GPU(&csrMatrix, X, Z);
//    SpMV_CSR_seq(&csrMatrix, X, Y);
//    vec_print((float *)Y, matrix.col_size);
//
//    return 0;
//}

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
    Vector X = Vector(csrMatrix.getRowSize());
    Vector Y = Vector(csrMatrix.getRowSize());
    Vector Z = Vector(csrMatrix.getRowSize());
    X.set(1.0f);
    Y.set(0.0f);
    Z.set(0.0f);
    csrMatrix.SpMV(X.getData(), Y.getData());
    csrMatrix.SpMV_GPU(X.getData(), Z.getData());
    if (Y.equals(Z)) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        std::cout << "Printing result of Sequential SpMV" << std::endl;
        Y.print();
        std::cout << std::endl;
        Z.print();
        std::cout << "Printing result of Parallel SpMV" << std::endl;
    }

}
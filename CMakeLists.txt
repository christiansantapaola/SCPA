cmake_minimum_required(VERSION 3.18)
project(SparseMatrix  LANGUAGES C CXX CUDA)
# find package
find_package(CUDA 11.0 REQUIRED)
set(CMAKE_BINARY_DIR bin)
include_directories(include)
include_directories(lib/mmio/include)

link_directories(lib)

# set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CUDA_ARCHITECTURES 52 75)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(spmvELL
        src/ellmatrixMain.cpp
        src/sparsematrix/CSRMatrix.cpp
        src/sparsematrix/COOMatrix.cpp
        src/spmv/SpMV_CSR.cu
        lib/mmio/src/mmio.c
        src/sparsematrix/Vector.cpp
        src/sparsematrix/ELLMatrix.cpp
        src/util/SwapMap.cpp
        src/util/Histogram.cpp
        src/util/cudaUtils.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.cpp)
target_link_libraries(spmvELL ${CUDA_LIBRARIES})

add_executable(spmvCSR
        src/csrMatrixMain.cpp
        src/sparsematrix/CSRMatrix.cpp
        src/sparsematrix/COOMatrix.cpp
        src/spmv/SpMV_CSR.cu
        lib/mmio/src/mmio.c
        src/sparsematrix/Vector.cpp
        src/sparsematrix/ELLMatrix.cpp
        src/util/SwapMap.cpp
        src/util/Histogram.cpp
        src/util/cudaUtils.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.cpp)


target_link_libraries(spmvCSR ${CUDA_LIBRARIES})

add_executable(readCSR
        src/readCSRMatrix.cpp
        src/sparsematrix/CSRMatrix.cpp
        src/sparsematrix/COOMatrix.cpp
        src/spmv/SpMV_CSR.cu
        lib/mmio/src/mmio.c
        src/sparsematrix/Vector.cpp
        src/sparsematrix/ELLMatrix.cpp
        src/util/SwapMap.cpp
        src/util/Histogram.cpp
        src/util/cudaUtils.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.cpp
        )

target_link_libraries(readCSR ${CUDA_LIBRARIES})

add_executable(readELL
        src/readELLMatrix.cpp
        src/sparsematrix/CSRMatrix.cpp
        src/sparsematrix/COOMatrix.cpp
        src/spmv/SpMV_CSR.cu
        lib/mmio/src/mmio.c
        src/sparsematrix/Vector.cpp
        src/sparsematrix/ELLMatrix.cpp
        src/util/SwapMap.cpp
        src/util/Histogram.cpp
        src/util/cudaUtils.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.cpp
        )
target_link_libraries(readELL ${CUDA_LIBRARIES})

cmake_minimum_required(VERSION 3.18)
project(SparseMatrix  LANGUAGES C CXX CUDA)
# find package
find_package(CUDA 11.0 REQUIRED)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(include)
include_directories(lib/mmio/include)

link_directories(lib)

# set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_CUDA_ARCHITECTURES 52 75)

# Add executable target with source files listed in SOURCE_FILES variable
#

add_executable(readCOO
        src/readCOOMatrix.c
        src/sparsematrix/COOMatrix.c
        lib/mmio/src/mmio.c)

add_executable(readCSR
        src/readCSRMatrix.c
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/COOMatrix.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c)

add_executable(readELL
        src/readELLMatrix.c
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/ELLMatrix.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c
)

add_executable(spmvCSR
        src/csrMatrixMain.c
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/CSRMatrix.cu
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/Vector.c
        src/sparsematrix/Vector.cu
        src/spmv/SpMV_CSR.cu
        lib/mmio/src/mmio.c
        src/util/Histogram.c
        src/util/cudaUtils.cu
        src/spmv/SpMVResult.c
        src/spmv/SpMV.c
        src/util/BlockGridInfo.c
        )

target_link_libraries(spmvCSR ${CUDA_LIBRARIES})
set_target_properties(spmvCSR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(spmvELL
        src/ellmatrixMain.c
        src/sparsematrix/ELLMatrix.c
        src/sparsematrix/ELLMatrix.cu
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/Vector.c
        src/sparsematrix/Vector.cu
        lib/mmio/src/mmio.c
        src/util/Histogram.c
        src/util/cudaUtils.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.c
        src/spmv/SpMV.c
        src/util/BlockGridInfo.c)

target_link_libraries(spmvELL ${CUDA_LIBRARIES})
set_target_properties(spmvELL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

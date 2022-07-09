cmake_minimum_required(VERSION 3.18)
project(SparseMatrix  LANGUAGES C CXX CUDA)
# find package
find_package(CUDA 11.0 REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -noeh")
set(CMAKE_CUDA_FLAGS_DEBUG "--profile --debug --device-debug -noeh")
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
        src/sparsematrix/MTXParser.c
        src/util/Histogram.c
        lib/mmio/src/mmio.c)

add_executable(readCSR
        src/readCSRMatrix.c
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/MTXParser.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c)

add_executable(readELL
        src/readELLMatrix.c
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/ELLMatrix.c
        src/sparsematrix/MTXParser.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c
)
target_link_libraries(readELL OpenMP::OpenMP_C)

add_executable(spmvTest
        src/spmvTest.c
        src/sparsematrix/ELLMatrix.c
        src/sparsematrix/ELLMatrix.cu
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/CSRMatrix.cu
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/Vector.c
        src/sparsematrix/Vector.cu
        src/sparsematrix/MTXParser.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c
        src/util/cudaUtils.cu
        src/spmv/SpMV_CSR.cu
        src/spmv/SpMV_ELLHYB.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.c
        src/spmv/SpMV.c
        src/spmv/SpMV_OMP.c
        src/spmv/SpMV_ELLHYB.c
        src/util/BlockGridInfo.c
        src/util/util.c)

target_link_libraries(spmvTest ${CUDA_LIBRARIES} OpenMP::OpenMP_C)


add_executable(spmvCSR
        src/spmvCSR.c
        src/sparsematrix/ELLMatrix.c
        src/sparsematrix/ELLMatrix.cu
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/CSRMatrix.cu
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/Vector.c
        src/sparsematrix/Vector.cu
        src/sparsematrix/MTXParser.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c
        src/util/cudaUtils.cu
        src/spmv/SpMV_CSR.cu
        src/spmv/SpMV_ELLHYB.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.c
        src/spmv/SpMV.c
        src/spmv/SpMV_OMP.c
        src/spmv/SpMV_ELLHYB.c
        src/util/BlockGridInfo.c
        src/util/util.c)

target_link_libraries(spmvCSR ${CUDA_LIBRARIES} OpenMP::OpenMP_C)
set_target_properties(spmvCSR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(spmvELL
        src/spmvELL.c
        src/sparsematrix/ELLMatrix.c
        src/sparsematrix/ELLMatrix.cu
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/CSRMatrix.cu
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/Vector.c
        src/sparsematrix/Vector.cu
        src/sparsematrix/MTXParser.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c
        src/util/cudaUtils.cu
        src/spmv/SpMV_ELLHYB.cu
        src/spmv/SpMV_ELL.cu
        src/spmv/SpMVResult.c
        src/spmv/SpMV.c
        src/spmv/SpMV_OMP.c
        src/spmv/SpMV_ELLHYB.c
        src/util/BlockGridInfo.c
        src/util/util.c)

target_link_libraries(spmvELL ${CUDA_LIBRARIES} OpenMP::OpenMP_C)
set_target_properties(spmvELL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(spmvCSR_OMP
        src/spmvCSR_OpenMP.c
        src/sparsematrix/ELLMatrix.c
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/Vector.c
        src/sparsematrix/MTXParser.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c
        src/spmv/SpMVResult.c
        src/spmv/SpMV.c
        src/spmv/SpMV_OMP.c
        src/util/util.c)

target_link_libraries(spmvCSR_OMP OpenMP::OpenMP_C)

add_executable(spmvELL_OMP
        src/spmvELL_OpenMP.c
        src/sparsematrix/ELLMatrix.c
        src/sparsematrix/CSRMatrix.c
        src/sparsematrix/COOMatrix.c
        src/sparsematrix/Vector.c
        src/sparsematrix/MTXParser.c
        lib/mmio/src/mmio.c
        src/util/Histogram.c
        src/spmv/SpMVResult.c
        src/spmv/SpMV.c
        src/spmv/SpMV_OMP.c
        src/util/BlockGridInfo.c
        src/util/util.c)

target_link_libraries(spmvELL_OMP OpenMP::OpenMP_C)

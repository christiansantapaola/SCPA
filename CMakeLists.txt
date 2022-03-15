cmake_minimum_required(VERSION 3.13)
project(SparseMatrix  LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES src/main.cpp src/CSRMatrix.cpp src/COOMatrix.cpp src/SpMV_CSR.cu lib/mmio/src/mmio.c src/Vector.cpp src/ELLMatrix.cpp)
set(CMAKE_BINARY_DIR bin)
include_directories(include)
include_directories(lib/mmio/include)
link_directories(lib)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(SparseMatrixExec ${SOURCE_FILES})


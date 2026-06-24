cmake_minimum_required(VERSION 3.10)
project(SegFormerB0)

# ===================== C++ STANDARD =====================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===================== BUILD TYPE =====================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ===================== COMPILER FLAGS =====================
# High-performance flags (important for ARM / Raspberry Pi)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -funroll-loops -fopenmp -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# ===================== OPENMP =====================
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

# ===================== OpenCV =====================
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ===================== OpenBLAS =====================
# Handle both Windows and Linux (Raspberry Pi)

if(WIN32)
    include_directories("C:/openblas/include")
    link_directories("C:/openblas/lib")
    set(BLAS_LIB openblas)
else()
    find_library(BLAS_LIB openblas)
    if(NOT BLAS_LIB)
        message(FATAL_ERROR "OpenBLAS not found!")
    endif()
endif()

# ===================== INCLUDE DIRECTORIES =====================
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Headerfiles)

# ===================== SOURCE FILES =====================
set(SOURCES
    main.cpp
    Src/preprocessing.cpp
    Src/weight_loader.cpp
    Src/patch_embedding.cpp
    Src/transformer_block.cpp
    Src/encoder.cpp
    Src/decoder.cpp
)

# ===================== HEADERS =====================
set(HEADERS
    Headerfiles/preprocessing.hpp
    Headerfiles/weight_loader.hpp
    Headerfiles/patch_embedding.hpp
    Headerfiles/transformer_block.hpp
    Headerfiles/encoder.hpp
    Headerfiles/decoder.hpp
    Headerfiles/visualization.hpp
)

# ===================== EXECUTABLE =====================
add_executable(segformer ${SOURCES} ${HEADERS})

# ===================== LINK LIBRARIES =====================
target_link_libraries(segformer
    ${OpenCV_LIBS}
    ${BLAS_LIB}
    OpenMP::OpenMP_CXX
)

# ===================== OUTPUT INFO =====================
message(STATUS "=== SegFormer B0 Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenBLAS library: ${BLAS_LIB}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Source files: ${SOURCES}")
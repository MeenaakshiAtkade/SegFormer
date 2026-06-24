cmake_minimum_required(VERSION 3.10)
project(SegFormerB0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for optimization and warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

# Find OpenCV (only used for preprocessing and image loading)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Headerfiles)

# Source files
set(SOURCES
    main.cpp
    Src/preprocessing.cpp
    Src/weight_loader.cpp
    Src/patch_embedding.cpp
    Src/transformer_block.cpp
    Src/encoder.cpp
    Src/decoder.cpp
)

# Header files (for IDE organization)
set(HEADERS
    Headerfiles/preprocessing.hpp
    Headerfiles/weight_loader.hpp
    Headerfiles/patch_embedding.hpp
    Headerfiles/transformer_block.hpp
    Headerfiles/encoder.hpp
    Headerfiles/decoder.hpp
    Headerfiles/visualization.hpp
)

# Create executable
add_executable(segformer ${SOURCES} ${HEADERS})

# Link OpenCV (only for preprocessing and image I/O)
target_link_libraries(segformer ${OpenCV_LIBS})

# Print configuration
message(STATUS "=== SegFormer B0 Configuration ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Source files: ${SOURCES}")

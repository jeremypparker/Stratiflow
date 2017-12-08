project(Stratiflow)
cmake_minimum_required(VERSION 3.1.3)

option(CUDA "Whether to enable CUDA acceleration of DFTs")

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(SYSTEM /home/jpp39/installs/include/eigen3/)
include_directories(SYSTEM /usr/include/eigen3/)
include_directories(SYSTEM /home/jpp39/installs/include)
include_directories(SYSTEM /usr/include/python2.7)
include_directories(SYSTEM external)
link_libraries(python2.7)
link_libraries(fftw3f fftw3f_omp m)

if(CUDA)
    link_libraries(cufft cudart)
    add_definitions(-DUSE_CUDA)
endif()

add_library(StratiLib Differentiation.cpp Field.cpp Graph.cpp)

add_subdirectory(tests)

add_executable(Stratiflow Stratiflow.cpp)
target_link_libraries(Stratiflow StratiLib)

add_executable(SnapshotViewer SnapshotViewer.cpp)
target_link_libraries(SnapshotViewer StratiLib)

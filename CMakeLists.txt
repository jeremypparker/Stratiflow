project(Stratiflow)
cmake_minimum_required(VERSION 3.1.3)

option(DOUBLE "Use double precision floating point")

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -g")

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
link_libraries(m)

if(DOUBLE)
    link_libraries(fftw3 fftw3_omp)
    add_definitions(-DUSE_DOUBLE)
else()
    link_libraries(fftw3f fftw3f_omp)
endif()

add_library(StratiLib Differentiation.cpp Field.cpp Graph.cpp Integration.cpp)

add_subdirectory(tests)

add_executable(DAL DAL.cpp)
target_link_libraries(DAL StratiLib)

add_executable(DNS DNS.cpp)
target_link_libraries(DNS StratiLib)

add_executable(SnapshotViewer SnapshotViewer.cpp)
target_link_libraries(SnapshotViewer StratiLib)

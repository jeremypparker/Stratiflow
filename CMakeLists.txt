project(Stratiflow)
cmake_minimum_required(VERSION 3.1.3)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(Threads)

include_directories(SYSTEM /usr/include/eigen3/)
include_directories(SYSTEM /home/jpp39/installs/include)
include_directories(SYSTEM /usr/include/python2.7)
include_directories(SYSTEM external)
link_libraries(python2.7)
link_libraries(fftw3f fftw3f_threads m)
link_libraries(cufft cudart)
link_libraries(${CMAKE_THREAD_LIBS_INIT})

add_library(StratiLib Differentiation.cpp Field.cpp Graph.cpp)

add_subdirectory(tests)

add_executable(Stratiflow Stratiflow.cpp)
target_link_libraries(Stratiflow StratiLib)

project(Stratiflow)
cmake_minimum_required(VERSION 3.1.3)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 11)

include_directories(SYSTEM /home/jpp39/installs/include/eigen3/)
include_directories(SYSTEM /usr/include/python2.7)
include_directories(SYSTEM external)
link_libraries(python2.7)
link_libraries(fftw3)

add_library(StratiLib Differentiation.cpp Field.cpp Field.cpp)

add_subdirectory(tests)

add_executable(Stratiflow Stratiflow.cpp)
target_link_libraries(Stratiflow StratiLib)

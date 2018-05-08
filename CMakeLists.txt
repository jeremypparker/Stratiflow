project(Stratiflow)
cmake_minimum_required(VERSION 3.1.3)

option(DOUBLE "Use double precision floating point")
option(MKL "Use intel math kernel library")
option(CUDA "Use CUDA for FFTs")
option(DEBUGPLOT "Plot full range of graphs")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(PythonLibs REQUIRED)
include_directories("${PYTHON_INCLUDE_DIRS}")
link_libraries("${PYTHON_LIBRARIES}")

find_path(NUMPY_PATH numpy/arrayobject.h
          HINTS "${PYTHON_INCLUDE_DIRS}")
include_directories("${NUMPY_PATH}")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(SYSTEM external)
link_libraries(m)

if(DEBUGPLOT)
    add_definitions(-DDEBUG_PLOT)
endif()

if(DOUBLE)
    if(NOT MKL AND NOT CUDA)
        link_libraries(fftw3 fftw3_omp)
    endif()
    add_definitions(-DUSE_DOUBLE)
else()
    if(NOT MKL AND NOT CUDA)
        link_libraries(fftw3f fftw3f_omp)
    endif()
endif()

if(MKL)
    include_directories(SYSTEM ${MKLROOT}/include)
    link_directories(${MKLROOT}/lib/intel64)
    link_libraries(mkl_intel_lp64 mkl_gnu_thread mkl_core iomp5 pthread dl)
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif()

if(CUDA)
    link_libraries(cufft cufftw)
    add_definitions(-DUSE_CUDA)
endif()

add_library(StratiLib
    Differentiation.cpp
    Field.cpp
    Graph.cpp
    Integration.cpp
    OSUtils.cpp
    FFT.cpp
    OrrSommerfeld.cpp
    Parameters.cpp
    StateVector.cpp
    IMEXRK.cpp)
target_link_libraries(StratiLib Eigen3::Eigen)

add_subdirectory(tests)

add_executable(DAL DAL.cpp)
target_link_libraries(DAL StratiLib)

add_executable(DNS DNS.cpp)
target_link_libraries(DNS StratiLib)

add_executable(OrrSommerfeld LargestGrowth.cpp)
target_link_libraries(OrrSommerfeld StratiLib)

add_executable(NewtonKrylov NewtonKrylov.cpp)
target_link_libraries(NewtonKrylov StratiLib)

add_executable(DALSteady DALSteady.cpp)
target_link_libraries(DALSteady StratiLib)

add_executable(FieldNorm FieldNorm.cpp)
target_link_libraries(FieldNorm StratiLib)

add_executable(PowerIterate PowerIterate.cpp)
target_link_libraries(PowerIterate StratiLib)

add_executable(PseudoArclength PseudoArclength.cpp)
target_link_libraries(PseudoArclength StratiLib)

add_executable(FindFold FindFold.cpp)
target_link_libraries(FindFold StratiLib)

add_executable(FindPitchfork FindPitchfork.cpp)
target_link_libraries(FindPitchfork StratiLib)

add_executable(TrackBifurcation TrackBifurcation.cpp)
target_link_libraries(TrackBifurcation StratiLib)


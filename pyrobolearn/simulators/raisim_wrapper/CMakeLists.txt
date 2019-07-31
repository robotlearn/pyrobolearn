# Run: cmake -DPYBIND11_PYTHON_VERSION=<PYTHON_VERSION> -DCMAKE_PREFIX_PATH=$LOCAL_BUILD ..
# where <PYTHON_VERSION>=2.7 or 3.*, LOCAL_BUILD is the build for raisimLib and raisimOgre

cmake_minimum_required(VERSION 2.8.9)
project (raisim_wrapper)

# find the various packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED eigen3)
# find_package(OpenMP REQUIRED)
find_package(raisim CONFIG REQUIRED)
find_package(raisimOgre CONFIG REQUIRED)

# header files
include_directories(include ${EIGEN3_INCLUDE_DIRS})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# source files
file(GLOB SOURCES "src/*.cpp")

pybind11_add_module(raisimpy ${SOURCES})
target_link_libraries(raisimpy raisim::raisim raisim::raisimOgre)

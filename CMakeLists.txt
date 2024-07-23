cmake_minimum_required(VERSION 3.10)

project(SGMM-MPI)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_C_FLAGS "${CMAKE_FLAGS} ${MPI_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS}")

add_executable(${CMAKE_PROJECT_NAME} main.cpp "utils.h")

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${MPI_LIBRARIES})

cmake_minimum_required(VERSION 3.15)
project(Benchmarks)

set(CMAKE_CXX_STANDARD 14)

add_executable(Benchmarks main.cpp Layer.h Weights.h NNUtils.h)
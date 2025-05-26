
cmake_minimum_required(VERSION 3.31)
project(pong)

set(CMAKE_CXX_STANDARD 20)

add_executable(epic1 epic1/tensor.h epic1/tests_epic1.cpp)
add_executable(epic2 epic2/layer.h epic2/dense.h epic2/activation.h epic2/loss.h epic2/neural_network.cpp epic2/tests_epic2.cpp)

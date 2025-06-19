cmake_minimum_required(VERSION 3.31)
project(pong)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(epic1
        epic1/tensor.h epic1/tests_epic1.cpp)
add_executable(epic2
        epic2/nn_dense.h epic2/nn_activation.h
        epic2/nn_loss.h epic2/tensor.h
        epic2/nn_interfaces.h epic2/nn_optimizer.h
        epic2/neural_network.h epic2/tests_epic2.cpp)

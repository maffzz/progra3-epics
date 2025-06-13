#ifndef PONG_NEURAL_NETWORK_H
#define PONG_NEURAL_NETWORK_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {
    template<typename T>
    class NeuralNetwork {
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {}

        template<template<typename ...> class LossType,
                template<typename ...> class OptimizerType = SGD>
        void train(const Tensor<T, 2> &X, const Tensor<T, 2> &Y,
                   const size_t epochs, const size_t batch_size, T learning_rate) {}

        // Para realizar predicciones
        Tensor<T, 2> predict(const Tensor<T, 2> &X) {}
    };
}

#endif //PONG_NEURAL_NETWORK_H
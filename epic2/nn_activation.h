#ifndef PONG_NN_ACTIVATION_H
#define PONG_NN_ACTIVATION_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {

    template<typename T>
    class ReLU final : public ILayer<T> {
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override {}
        Tensor<T,2> backward(const Tensor<T,2>& g) override {}
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override {}
        Tensor<T,2> backward(const Tensor<T,2>& g) override {}
    };
}

#endif //PONG_NN_ACTIVATION_H
#ifndef PONG_NN_OPTIMIZER_H
#define PONG_NN_OPTIMIZER_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {
    template<typename T>
    class SGD final : public IOptimizer<T> {
    public:
        explicit SGD(T learning_rate = 0.01) {}
        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {}
    };
}

#endif //PONG_NN_OPTIMIZER_H
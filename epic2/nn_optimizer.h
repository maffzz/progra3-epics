#ifndef PONG_NN_OPTIMIZER_H
#define PONG_NN_OPTIMIZER_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {

    template<typename T>
    class SGD final : public IOptimizer<T> {
    private:
        T learning_rate;

    public:
        explicit SGD(T learning_rate = 0.01) : learning_rate(learning_rate) {}

        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override {
            for (size_t i = 0; i < params.shape()[0]; ++i) {
                for (size_t j = 0; j < params.shape()[1]; ++j) {
                    params(i, j) -= learning_rate * grads(i, j);}}}};

}

#endif //PONG_NN_OPTIMIZER_H
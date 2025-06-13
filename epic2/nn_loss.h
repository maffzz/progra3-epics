#ifndef PONG_NN_LOSS_H
#define PONG_NN_LOSS_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {
    template<typename T>
    class MSELoss final: public ILoss<T, 2> {
    public:
        MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) {}
        T loss() const override {}
        Tensor<T,2> loss_gradient() const override {}
    };

    template<typename T>
    class BCELoss final: public ILoss<T, 2> {
    public:
        BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) {}
        T loss() const override {}
        Tensor<T,2> loss_gradient() const override {}
    };
}

#endif //PONG_NN_LOSS_H
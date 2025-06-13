#ifndef PONG_NN_DENSE_H
#define PONG_NN_DENSE_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T> {
    public:
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) {}
        Tensor<T,2> forward(const Tensor<T,2>& x) override {}
        Tensor<T,2> backward(const Tensor<T,2>& dZ) override {}
        void update_params(IOptimizer<T>& optimizer) override {}
    };
}

#endif //PONG_NN_DENSE_H
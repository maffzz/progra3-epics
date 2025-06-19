
#include "../epic1/tensor.h"
#include "layer.h"
using namespace utec::algebra;

namespace utec::neural_network {
    template <typename T>
    class ReLU : public ILayer <T> {
        Tensor<T,2> mask; // para almacenar dónde x>0
    public:
        Tensor<T,2> forward(const Tensor<T,2>& x) override {};
        Tensor<T,2> backward(const Tensor<T,2>& grad) override {};
    };
}
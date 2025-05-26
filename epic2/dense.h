
#include "../epic1/tensor.h"
#include "layer.h"
using namespace utec::algebra;

namespace utec::neural_network {
    template <typename T>
    class Dense : public ILayer <T> {
        Tensor<T,2> W, dW; // [in , out] y su gradiente
        Tensor<T,1> b, db; // [ out] y su gradiente
        Tensor<T,2> last_x; // cache de entrada para backward
    public:
        Dense(size_t in_feats, size_t out_feats) {};
        Tensor<T,2> forward(const Tensor<T,2>& x) override {};
        Tensor<T,2> backward(const Tensor<T,2>& grad) override {};
    };
}

#include "../epic1/tensor.h"
using namespace utec::algebra;

namespace utec::neural_network {
    template <typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;
        // Forward: recibe batch x features, devuelve batch x units
        virtual Tensor<T,2> forward(const Tensor<T,2>& x) = 0;
        // Backward: recibe gradiente de salida, devuelve gradiente de entrada
        virtual Tensor<T,2> backward(const Tensor<T,2>& grad) = 0;
    };
}
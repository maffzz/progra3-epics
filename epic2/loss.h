
#include "../epic1/tensor.h"
using namespace utec::algebra;

namespace utec::neural_network {
    template <typename T>
    class MSELoss {
        Tensor<T,2> last_pred, last_target;
    public:
        // Devuelve la pérdida media
        T forward(const Tensor<T,2>& pred, const Tensor<T,2>& target) {};
        // Devuelve dL/dpred
        Tensor<T,2> backward() {};
    };
}
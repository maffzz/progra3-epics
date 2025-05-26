
#include "layer.h"
#include "loss.h"
#include <vector>

namespace utec::neural_network {
    template <typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        MSELoss<T> criterion;
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {};
        // Ejecuta forward por todas las capas
        Tensor<T,2> forward(const Tensor<T,2>& x) {};
        // Lanza backward desde la ú ltima capa
        void backward(const Tensor<T,2>& grad) {};
        // Actualiza todos los pará metros con learning rate lr
        void optimize(T lr) {};
        // Entrena con X, Y durante epochs
        void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, size_t epochs, T lr) {};
    };
}
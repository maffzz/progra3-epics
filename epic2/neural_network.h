#ifndef PONG_NEURAL_NETWORK_H
#define PONG_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include "nn_interfaces.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
using namespace utec::algebra;

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers.push_back(std::move(layer));}

        template <template <typename> class LossType, template <typename> class OptimizerType = SGD>
        void train(const Tensor<T, 2>& X, const Tensor<T, 2>& Y, const size_t epochs, const size_t batch_size, T learning_rate) {
            OptimizerType<T> optimizer(learning_rate);

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                Tensor<T, 2> Y_pred = predict(X);
                LossType<T> loss(Y_pred, Y);

                Tensor<T, 2> dL = loss.loss_gradient();

                for (int i = layers.size() - 1; i >= 0; --i) {
                    dL = layers[i]->backward(dL);}

                for (auto& layer : layers) {
                    layer->update_params(optimizer);}

                if (epoch % 100 == 0) {
                    std::cout << "Epoch " << epoch << " Loss: " << loss.loss() << std::endl;}}}

        Tensor<T, 2> predict(const Tensor<T,2>& X) {
            Tensor<T, 2> result = X;
            for (auto& layer : layers) {
                result = layer->forward(result);}
            return result;}};

}

#endif //PONG_NEURAL_NETWORK_H
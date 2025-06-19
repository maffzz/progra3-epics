#ifndef PONG_NN_DENSE_H
#define PONG_NN_DENSE_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {

    template<typename T>
    class Dense final : public ILayer<T> {
    private:
        Tensor<T, 2> weights;  // Pesos de la capa
        Tensor<T, 2> biases;   // Sesgos de la capa
        Tensor<T, 2> input_cache;  // Para almacenar las entradas durante el forward

    public:
        Dense(size_t in_f, size_t out_f, auto init_w_fun, auto init_b_fun) {
            // Inicialización de los pesos con dimensiones (out_f, in_f)
            weights = Tensor<T, 2>({out_f, in_f});  // Tensor de pesos: out_f filas, in_f columnas
            init_w_fun(weights);  // Inicialización de los pesos usando la función pasada

            // Inicialización de los sesgos con dimensiones (out_f, 1)
            biases = Tensor<T, 2>({out_f, 1});  // Tensor de sesgos: out_f filas, 1 columna
            init_b_fun(biases);}

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            input_cache = x;  // Guardamos las entradas para la retropropagación
            // X * W^T + b (usamos la transposición de X)
            Tensor<T, 2> result = weights * x.T() + biases;
            return result.T();}

        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
            Tensor<T, 2> dW = dZ * input_cache.T();
            Tensor<T, 2> db = dZ.sum(0);
            return weights.T() * dZ;}

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(weights, weights);  // Actualizamos los pesos
            optimizer.update(biases, biases);}};

}

#endif //PONG_NN_DENSE_H
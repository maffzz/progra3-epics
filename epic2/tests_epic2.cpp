#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include "tensor.h"
#include "nn_activation.h"
#include "nn_dense.h"
#include "neural_network.h"
using namespace utec::algebra;

int main() {
    // Datos del problema XOR
    utec::algebra::Tensor<double, 2> X{{0, 0}, {0, 1}, {1, 0}, {1, 1}};  // Inicialización con valores
    utec::algebra::Tensor<double, 2> Y{{0}, {1}, {1}, {0}};  // Las salidas esperadas para XOR

    // Inicializador Xavier para pesos y sesgos
    std::mt19937 gen(42);
    auto xavier_init = [&](auto& parameter) {
        const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for (auto& v : parameter) v = dist(gen);};

    // Crear la red neuronal con capas y optimizador
    utec::neural_network::NeuralNetwork<double> net;
    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(2, 4, xavier_init, xavier_init));  // Capa oculta
    net.add_layer(std::make_unique<utec::neural_network::Sigmoid<double>>());  // Función de activación
    net.add_layer(std::make_unique<utec::neural_network::Dense<double>>(4, 1, xavier_init, xavier_init));  // Capa de salida
    net.add_layer(std::make_unique<utec::neural_network::Sigmoid<double>>());  // Función de activación

    // Entrenamiento con optimizador SGD
    constexpr size_t epochs = 4000;
    constexpr double learning_rate = 0.08;
    net.train<utec::neural_network::BCELoss, utec::neural_network::SGD>(X, Y, epochs, 4, learning_rate);

    // Predicción
    utec::algebra::Tensor<double, 2> Y_pred = net.predict(X);

    // Mostrar resultados
    for (size_t i = 0; i < 4; ++i) {
        const double p = Y_pred(i, 0);
        std::cout
                << std::fixed << std::setprecision(0)
                << "Input: (" << X(i, 0) << ", " << X(i, 1) << ") -> Prediction: " << p << std::endl;}

    return 0;}
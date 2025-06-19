#ifndef PONG_NN_ACTIVATION_H
#define PONG_NN_ACTIVATION_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {

    template<typename T>
    class ReLU final : public ILayer<T> {
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override {
            Tensor<T,2> resultado(z.shape());
            for (size_t i = 0; i < z.shape()[0]; ++i) {
                for (size_t j = 0; j < z.shape()[1]; ++j) {
                    resultado(i, j) = std::max(T(0), z(i, j));}}
            return resultado;}

        Tensor<T,2> backward(const Tensor<T,2>& g) override {
            Tensor<T,2> resultado(g.shape());
            for (size_t i = 0; i < g.shape()[0]; ++i) {
                for (size_t j = 0; j < g.shape()[1]; ++j) {
                    resultado(i, j) = (g(i, j) > T(0)) ? g(i, j) : T(0);}}
            return resultado;}};

    template<typename T>
    class Sigmoid final : public ILayer<T> {
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override {
            Tensor<T,2> resultado(z.shape());
            for (size_t i = 0; i < z.shape()[0]; ++i) {
                for (size_t j = 0; j < z.shape()[1]; ++j) {
                    resultado(i, j) = 1 / (1 + std::exp(-z(i, j)));}}
            return resultado;}

        Tensor<T,2> backward(const Tensor<T,2>& g) override {
            Tensor<T,2> resultado(g.shape());
            for (size_t i = 0; i < g.shape()[0]; ++i) {
                for (size_t j = 0; j < g.shape()[1]; ++j) {
                    T s = 1 / (1 + std::exp(-g(i, j)));
                    resultado(i, j) = s * (1 - s);}}
            return resultado;}};

}

#endif //PONG_NN_ACTIVATION_H
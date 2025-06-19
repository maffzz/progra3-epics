#ifndef PONG_NN_LOSS_H
#define PONG_NN_LOSS_H

#include "nn_interfaces.h"
using namespace utec::algebra;

namespace utec::neural_network {

    template<typename T>
    class MSELoss final : public ILoss<T, 2> {
        Tensor<T, 2> y_pred;
        Tensor<T, 2> y_true;
    public:
        MSELoss(const Tensor<T, 2>& y_prediction, const Tensor<T, 2>& y_true)
                : y_pred(y_prediction), y_true(y_true) {}

        T loss() const override {
            T sum = 0;
            for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                    T diff = y_pred(i, j) - y_true(i, j);
                    sum += diff * diff;}}
            return sum / static_cast<T>(y_pred.size());}

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> gradient(y_pred.shape());
            for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                    gradient(i, j) = 2 * (y_pred(i, j) - y_true(i, j)) / static_cast<T>(y_pred.size());}}
            return gradient;}};

    template<typename T>
    class BCELoss final : public ILoss<T, 2> {
        Tensor<T, 2> y_pred;
        Tensor<T, 2> y_true;
    public:
        BCELoss(const Tensor<T, 2>& y_prediction, const Tensor<T, 2>& y_true)
                : y_pred(y_prediction), y_true(y_true) {}

        T loss() const override {
            T sum = 0;
            for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                    T pred = y_pred(i, j);
                    T true_val = y_true(i, j);
                    sum -= true_val * std::log(pred) + (T(1) - true_val) * std::log(T(1) - pred);}}
            return sum / static_cast<T>(y_pred.size());}

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> gradient(y_pred.shape());
            for (size_t i = 0; i < y_pred.shape()[0]; ++i) {
                for (size_t j = 0; j < y_pred.shape()[1]; ++j) {
                    T pred = y_pred(i, j);
                    T true_val = y_true(i, j);
                    gradient(i, j) = (pred - true_val) / (pred * (T(1) - pred));}}
            return gradient;}};

}

#endif //PONG_NN_LOSS_H
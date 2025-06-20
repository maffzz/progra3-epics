# epic 2: proyecto final - neural network
**curso:** progra3  
**tarea:** proyecto final  
**cmake project:** prog3_nn_final_project_v2025_01

## question #1 - activation - ReLu y Sigmoid (2 points)

```c++
template<typename G>
class ReLU final : public ILayer<G> {
public:
    Tensor<G,2> forward(const Tensor<G,2>& z) override { ... }
    Tensor<G,2> backward(const Tensor<G,2>& g) override { ... }
};
```
```c++
template<typename G>
class Sigmoid final : public ILayer<G> {
public:
    Tensor<G,2> forward(const Tensor<G,2>& z) override { ... }
    Tensor<G,2> backward(const Tensor<G,2>& g) override { ... }
};
```
**use case: ReLu**  
```cpp
using G = float;
auto relu = utec::neural_network::ReLU<G>();
// tensores
Tensor<G, 2> M({2,2}); M = {-1, 2, 0, -3};
Tensor<G, 2> GR({2,2}); GR.fill(1.0f);
// forward
auto R = relu.forward(M);
std::cout << R(0,1) << "\n"; // espera 2
// backward
const auto dM = relu.backward(GR);
std::cout << dM;
```
**use case: Sigmoid**
```c++
auto sigmoid = utec::neural_network::Sigmoid<G>();
// tensores
constexpr int rows = 5;
constexpr int cols = 4;
Tensor<G, 2> M({rows, cols});
M.fill(-100.0);
for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
        if (i == j) M(i, j) = 100.0;
        if (i == rows - 1 - j) M(i, j) = 100.0;}
std::cout << std::fixed << std::setprecision(1);
std::cout << M << std::endl;
// forward
const auto S = sigmoid.forward(M);
std::cout << S << std::endl;
// backward
Tensor<G, 2> GR({rows,cols}); GR.fill(1.0);
const auto dM = sigmoid.backward(GR);
std::cout << dM << std::endl;
```

## question #2 - loss function - MSE y BCE (2 points)

```c++
template<typename G>
class MSELoss final: public ILoss<G, 2> {
public:
    MSELoss(const Tensor<G,2>& y_prediction, const Tensor<G,2>& y_true) { ... }
    G loss() const override { ... }
    Tensor<G,2> loss_gradient() const override { ... }
};
```
```c++
template<typename G>
class BCELoss final: public ILoss<G, 2> {
public:
    BCELoss(const Tensor<G,2>& y_prediction, const Tensor<G,2>& y_true) { ... }
    G loss() const override { ... }
    Tensor<G,2> loss_gradient() const override { ... }
};
```
**use case: MSE**  
```cpp
using G = double;
// tensores
Tensor<G,2> y_predicted({1,2}); y_predicted = {1, 2};
Tensor<G,2> y_expected({1,2}); y_expected = {0, 4};
const utec::neural_network::MSELoss<G> mse_loss(y_predicted, y_expected);
// forward
const G loss = mse_loss.loss();
std::cout << loss << "\n";
// backward
const Tensor<G,2> dP = mse_loss.loss_gradient();
std::cout << dP;
```
**use case: BCE**
```c++
using G = double;
// tensores
Tensor<G,2> y_predicted({1,2}); y_predicted = {0.9, 0.1};
Tensor<G,2> y_expected({1,2}); y_expected = {0, 1};

const utec::neural_network::BCELoss<G> bce_loss(y_predicted, y_expected);
// forward
const G loss = bce_loss.loss();
std::cout << loss << "\n";
// backward
const Tensor<G,2> dP = bce_loss.loss_gradient();
std::cout << dP;
```

## question #3 - dense layer (6 points)

```c++
template<typename G>
class Dense final : public ILayer<G> {
public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) { ... }
    Tensor<G,2> forward(const Tensor<G,2>& x) override { ... }
    Tensor<G,2> backward(const Tensor<G,2>& dZ) override { ... }
    void update_params(IOptimizer<G>& optimizer) override { ... }
};
```

**use case: using identity initializer and zero**
```cpp
using G = double;

// inicializador identidad
auto init_identity = [](Tensor<G,2>& M) {
    const auto shape = M.shape();
    const size_t rows = shape[0];
    const size_t cols = shape[1];
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            M(i,j) = (i == j ? 1.0 : 0.0);
};

// inicializador de ceros
auto init_zero = [](Tensor<G,2>& M) {
    for (auto& v : M) v = 0.0;
};
constexpr int n_batches = 2;
constexpr int in_features = 4;
constexpr int out_features = 3;
Dense<double> layer(size_t{in_features}, size_t{out_features},init_identity, init_zero);

Tensor<G,2> X1({n_batches, in_features});
std::iota(X1.begin(), X1.end(), 1);
// forward
Tensor<G,2> Y = layer.forward(X1);
std::cout << Y << std::endl;

Tensor<G,2> Z({n_batches, out_features});
std::iota(Z.begin(), Z.end(), 1);
auto Z_adjusted = Z / static_cast<G>(Z.size());

Tensor<G,2> X_adjusted = layer.backward(Z_adjusted);
// X ajustado
std::cout << X_adjusted << std::endl;
```
**use case: using Xavier initializer**
```c++
using G = double;

// inicializador Xavier
std::mt19937 gen(4);
auto xavier_init = [&](auto& parameter) {
    const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
    std::uniform_real_distribution<> dist(-limit, limit);
    for (auto& v : parameter) v = dist(gen);
};

constexpr int n_batches = 2;
constexpr int in_features = 4;
constexpr int out_features = 3;
Dense<double> layer(size_t{in_features}, size_t{out_features},xavier_init, xavier_init);

Tensor<G,2> X1({n_batches, in_features});
std::iota(X1.begin(), X1.end(), 1);
// forward
Tensor<G,2> Y = layer.forward(X1);
std::cout << Y << std::endl;

Tensor<G,2> Z({n_batches, out_features});
std::iota(Z.begin(), Z.end(), 1);
auto Z_adjusted = Z / static_cast<G>(Z.size());

Tensor<G,2> X_adjusted = layer.backward(Z_adjusted);
// X ajustado
std::cout << X_adjusted << std::endl;
```

## question #4 - entrenamiento - XOR (8 points)
  
```c++
template<typename G>
class NeuralNetwork {
public:
    void add_layer(std::unique_ptr<ILayer<G>> layer) { ...}
    template <template <typename ...> class LossType, 
        template <typename ...> class OptimizerType = SGD>
    void train( const Tensor<G,2>& X,const Tensor<G,2>& Y, 
        const size_t epochs, const size_t batch_size, G learning_rate) { ... }
    // para realizar predicciones
    Tensor<G,2> predict(const Tensor<G,2>& X) { ... }
};
```

**use case: using He initializer y MSE loss function**
```c++
constexpr size_t batch_size = 4;
Tensor<double,2> X({batch_size, 2});
Tensor<double,2> Y({batch_size, 1});

// datos XOR
X = { 0, 0,
      0, 1,
      1, 0,
      1, 1};
Y = { 0, 1, 1, 0};

// inicializador He
std::mt19937 gen(42);

auto init_he = [&](Tensor<double,2>& M) {
    const double last = 2.0/(static_cast<double>(M.shape()[0]+ M.shape()[1]));
    std::normal_distribution<double> dist(
        0.0,
        std::sqrt(last));
    for (auto& v : M) v = dist(gen);
};

// construcción de la red
NeuralNetwork<double> net;
net.add_layer(std::make_unique<Dense<double>>(
    size_t{2}, size_t{4}, init_he, init_he));
net.add_layer(std::make_unique<ReLU<double>>());
net.add_layer(std::make_unique<Dense<double>>(
    size_t{4}, size_t{1}, init_he, init_he));

// entrenamiento
constexpr size_t epochs = 3000;
constexpr double learning_rate = 0.08;
net.train<MSELoss> (X, Y, epochs, batch_size, learning_rate);

// predicción
Tensor<double,2> Y_prediction = net.predict(X);

// verificación
for (size_t i = 0; i < batch_size; ++i) {
    const double p = Y_prediction(i,0);
    std::cout
        << std::fixed << std::setprecision(0)
        << "input: (" << X(i,0) << "," << X(i,1)
        << std::fixed << std::setprecision(4)
        << ") -> prediction: " << p << std::endl;
    if (Y(i,0) < 0.5) {
        assert(p < 0.5); // expected output close to 0
    } else {
        assert(p >= 0.6); // expected output close to 1
    }
}
```
**use case: using Xavier initializer y BCE loss function**
```c++
constexpr size_t batch_size = 4;
Tensor<double,2> X({batch_size, 2});
Tensor<double,2> Y({batch_size, 1});

// datos XOR
X = { 1, 0,
      0, 1,
      0, 0,
      1, 1};    
Y = { 1, 1, 0, 0};

// inicializador Xavier 
std::mt19937 gen(4);
auto xavier_init = [&](auto& parameter) {
    const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
    std::uniform_real_distribution<> dist(-limit, limit);
    for (auto& v : parameter) v = dist(gen);
};

// construcción de la red
NeuralNetwork<double> net;
net.add_layer(std::make_unique<Dense<double>>(
        size_t{2}, size_t{4}, xavier_init, xavier_init));
net.add_layer(std::make_unique<Sigmoid<double>>());
net.add_layer(std::make_unique<Dense<double>>(
        size_t{4}, size_t{1}, xavier_init, xavier_init));
net.add_layer(std::make_unique<Sigmoid<double>>());

// entrenamiento
constexpr size_t epochs = 4000;
constexpr double lr = 0.08;
net.train<BCELoss>(X, Y, epochs, batch_size, lr);

// predicción
Tensor<double,2> Y_prediction = net.predict(X);

// verificación
for (size_t i = 0; i < batch_size; ++i) {
    const double p = Y_prediction(i, 0);
    std::cout
        << std::fixed << std::setprecision(0)
        << "input: (" << X(i,0) << "," << X(i,1)
        << std::fixed << std::setprecision(4)
        << ") -> prediction: " << p << std::endl;
    if (Y(i,0) < 0.5) {
        assert(p < 0.5);} // expected output close to 0
    else {
        assert(p >= 0.6);}} // expected output close to 1
```

## question #5 - optimización (SGD y Adam) (2 points)

```c++
template<typename G>
class SGD final : public IOptimizer<G> {
public:
    explicit SGD(G learning_rate = 0.01) { ... }
    void update(Tensor<G, 2>& params, const Tensor<G, 2>& grads) override { ... }
};
```
```c++
template<typename G>
class Adam final : public IOptimizer<G> {
public:
    explicit Adam(G learning_rate = 0.001, G beta1 = 0.9, G beta2 = 0.999, G epsilon = 1e-8) { ... }
    void update(Tensor<G, 2>& params, const Tensor<G, 2>& grads) override { ... }
    void step() override { ... }
};
 ```
**use case: SGD**  
```c++
using G = float;

Tensor<G,2> W({2,2}); W.fill(1.0f);
Tensor<G,2> dW({2,2}); dW.fill(0.5f);
utec::neural_network::SGD<G> opt(0.1f);

opt.update(W, dW);
std::cout
    << std::fixed << std::setprecision(6)
    << W(0,0) << "\n";
```
**use case: Adam**
```c++
using G = float;

Tensor<G,2> W({20,25}); W.fill(1.0f);
Tensor<G,2> dW({20,25}); dW.fill(0.2f);
utec::neural_network::Adam opt(0.01f, 0.009f, 9.00f);

opt.update(W, dW);
std::cout << W(0,0) << "\n";
```
# epic 1: proyecto final - tensor  
**curso:** progra3  
**tarea:** proyecto final  
**cmake project:** prog3_tensor_final_project_v2025_01

## indicaciones
cada pregunta deberá incluir un archivo cabecera (.h) correspondiente:
 - `tensor.h`

## question #1 - creación, acceso, fill e impresión (3 points)
**use case #1:**
```cpp
const utec::algebra::Tensor<int, 2> t(2, 2);
constexpr std::array<size_t, 2> a1 = {2, 2};
const auto a2 = t.shape();
for (size_t i = 0; i < a1.size(); ++i) {
    REQUIRE(a1[i] == a2[i]);}
```
**use case #2:**
```cpp
    try {
        const utec::algebra::Tensor<int, 2> t(2, 2, 2);}
    catch (const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Number of dimensions do not match with 2
```
**use case #3:**
```cpp
    utec::algebra::Tensor<int, 3> t(2, 2, 3);
    t.fill(117);
    std::cout << t;
```
**use case #4:**
```cpp
    utec::algebra::Tensor<int, 2> t(6, 2);
    try {
        t = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11};}
    catch (const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Data size does not match tensor size
```

## question #2 - reshape (3 points)

**use case #1:**
```cpp
    utec::algebra::Tensor<int, 2> t(6, 2);
    t = {
        111, 222, 333, 444, 555, 666,
        777, 888, 999, 111, 111, 112};
    std::cout << t << '\n';
    t.reshape(2, 6);
    std::cout << t;
```
**use case #2:**
```cpp
    utec::algebra::Tensor<int, 2> t(6, 2);
    try {
        t = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12};
        t.reshape(3, 2, 2);}
    catch (const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Number of dimensions do not match with 2
```
**use case #3:**
```cpp
    utec::algebra::Tensor<int, 2> t(6, 2);
    t = {
        111, 222, 333, 444, 555, 666,
        777, 888, 999, 111, 111, 112};
    std::cout << t << '\n';
    t.reshape(2, 3); // valores validos, tensor disminuye de tamaño
    std::cout << t;
```
**use case #4:**
```cpp
    utec::algebra::Tensor<int, 2> t(6, 2);
    t = {
        100, 200, 300, 40, 500, 60,
        700, 800, 90, 1000, 110, 1200};
    std::cout << t << '\n';
    t.reshape(12, 1);
    std::cout << t << '\n';
    t.reshape(1, 20);
    std::cout << t;
```

## question #3 - suma y resta de tensores (2 points)

**use case #1:**
```cpp
    utec::algebra::Tensor<double, 2> a(8, 6), b(8, 6);
    a(1, 2) = 2.5;
    a(7, 5) = 10.5;
    b.fill(21.1);
    const auto sum = a + b;
    std::cout << sum << std::endl;
```
**use case #2:**
```cpp
    utec::algebra::Tensor<double, 2> a(8, 6), b(8, 6);
    a.fill(100.0);
    a(1, 2) += -2.5;
    a(7, 5) += -10.5;
    b.fill(21.1);
    const auto sum = a - b;
    std::cout << sum << std::endl;
```
**use case #3:**
```cpp
    utec::algebra::Tensor<double, 2> t1(20, 50), t2(50, 20);
    t1.fill(1);
    t2.fill(4);
    try {
        auto result = t1 + t2;}
    catch(const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Shapes do not match, and they are not compatible for broadcasting
```
**use case #4:**
```cpp
    utec::algebra::Tensor<double, 3> t1(200, 50, 3), t2(200, 50, 3), t3(200, 50, 3);
    t1.fill(3);
    t2.fill(4);
    t3.fill(5);
    const auto r = t1 + t2 - t3;
    std::cout << std::accumulate(r.cbegin(), r.cend(), double{0});
```

## question #4 - multiplicación y operaciones con escalares (2 points)

**use case #1:**
```cpp
    utec::algebra::Tensor<int, 3> t1(3, 4, 5), t2(3, 4, 5), t3(3, 4, 5);
    std::iota(t1.begin(), t1.end(), 5);
    std::iota(t2.begin(), t2.end(), 10);
    std::iota(t3.begin(), t3.end(), 100);
    auto r = t1 * t2 + t3;
    std::cout << r;
```
**use case #2:**
```cpp
    utec::algebra::Tensor<int, 3> t1(3, 4, 5), t2(3, 5, 4), t3(4, 3, 5);
    std::iota(t1.begin(), t1.end(), 5);
    std::iota(t2.begin(), t2.end(), 10);
    std::iota(t3.begin(), t3.end(), 100);
    try {
        const auto r = t1 * t2 + t3;
        std::cout << r;}
    catch (const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Shapes do not match, and they are not compatible for broadcasting
```
**use case #3:**
```cpp
    utec::algebra::Tensor<int, 3> t1(3, 4, 5), t2(3, 5, 4), t3(4, 3, 5);
    std::iota(t1.begin(), t1.end(), 5);
    std::iota(t2.begin(), t2.end(), 10);
    std::iota(t3.begin(), t3.end(), 100);
    t2.reshape(3, 4, 5);
    t3.reshape(3, 4, 5);
    const auto r = t1 * t2 - 5;
    std::cout << r;
```
**use case #4:**
```cpp
    utec::algebra::Tensor<int, 3> t1(3, 4, 5);
    std::iota(t1.begin(), t1.end(), 10);
    const auto r = 50 + (t1 + 90)/10;
    std::cout << r;
```

## question #5 - broadcasting implícito (2 points)

**use case #1:**
```cpp
    utec::algebra::Tensor<int, 2> t1(3, 4);
    t1.fill(1);
    utec::algebra::Tensor<int, 2> t2(3, 1); // aplicar Broadcasting a este vector
    t2.fill(5);
    const auto r = t1 * t2;
    std::cout << r;
```
**use case #2:**
```cpp
    utec::algebra::Tensor<int, 3> t1(3, 4, 5);
    t1.fill(11);
    utec::algebra::Tensor<int, 3> t2(1, 1, 5); // aplicar Broadcasting a este vector
    t2.fill(51);
    const auto r = t1 * t2;
    std::cout << r;
```
**use case #3:**
```cpp
    utec::algebra::Tensor<int, 3> t1(3, 4, 5), t2(3, 1, 5);
    std::iota(t1.begin(), t1.end(), 100);
    std::iota(t2.begin(), t2.end(), 10);
    const auto rest = t1 - t2;
    std::cout << rest;
```
**use case #4:**
```cpp
    utec::algebra::Tensor<double, 2> t1(6, 7), t2(6, 1);
    std::iota(t1.begin(), t1.end(), 10);
    t2.fill(100);
    const auto sum = t1 + t2;
    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << sum << std::endl;
```

## question #7 - transpose 2D (2 points)

**use case #1:**
```cpp
    utec::algebra::Tensor<int, 3> t1(2, 4, 3);
    std::iota(t1.begin(), t1.end(), 10);
    std::cout << t1 << '\n';
    const auto r = transpose_2d(t1);
    std::cout << r;
```
**use case #2:**
```cpp
    utec::algebra::Tensor<int, 2> t1(19, 24);
    std::iota(t1.begin(), t1.end(), 1);
    std::cout << t1 << '\n';
    const auto r = transpose_2d(t1);
    std::cout << r;
```
**use case #3:**
```cpp
    utec::algebra::Tensor<int, 1> t1(30);
    std::iota(t1.begin(), t1.end(), 1);
    std::cout << t1 << '\n';
    try {
        const auto r = transpose_2d(t1);}
    catch (const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Cannot transpose 1D tensor: need at least 2 dimensions
```
**use case #4:**
```cpp
    utec::algebra::Tensor<double, 4> t1(2, 2, 2, 3);
    std::iota(t1.begin(), t1.end(), 10.5);
    std::cout << t1 << '\n';
    const auto r = transpose_2d(t1);
    std::cout << r;
```

## question #7 - multiplicación matriz (4 points)

**use case #1:**
```cpp
    utec::algebra::Tensor<double, 3> t1(1, 5, 3);
    utec::algebra::Tensor<double, 3> t2(4, 3, 2);
    t1.fill(10);
    t2.fill(5);
    try {
        const auto r = utec::algebra::matrix_product(t1, t2);}
    catch (const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match
```
**use case #2:**
```cpp
    utec::algebra::Tensor<double, 2> t1(5, 3);
    utec::algebra::Tensor<double, 2> t2(5, 2);
    t1.fill(10);
    t2.fill(5);
    try {
        const auto r = utec::algebra::matrix_product(t1, t2);}
    catch (const exception& e) {
        std::cout << e.what();} // mensaje de ERROR: Matrix dimensions are incompatible for multiplication
```
**use case #3:**
```cpp
    utec::algebra::Tensor<int, 3> t1(2, 5, 3);
    utec::algebra::Tensor<int, 3> t2(2, 3, 7);
    std::iota(t1.begin(), t1.end(), 10);
    std::iota(t2.begin(), t2.end(), 5);
    const auto r = utec::algebra::matrix_product(t1, t2);
    std::cout << r;
```
**use case #4:**
```cpp
    utec::algebra::Tensor<int, 3> t1(2, 3, 2);
    utec::algebra::Tensor<int, 3> t2(2, 2, 4);
    std::iota(t1.begin(), t1.end(), 1);
    std::iota(t2.begin(), t2.end(), 3);
    const auto r = utec::algebra::matrix_product(t1, t2);
    std::cout << t1 << '\n';
    std::cout << t2 << '\n';
    std::cout << r;
```

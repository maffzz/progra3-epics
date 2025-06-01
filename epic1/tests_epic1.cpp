#include "tensor.h"
#include <iostream>
using namespace utec::algebra;

int main() {

    // test 1: creación de tensores de diferentes dimensiones
    std::cout << std:: endl << "test 1: creación de tensores"  << std::endl;
    Tensor<int, 1> t1(5);  // tensor 1D
    Tensor<int, 2> t2(3, 4);  // tensor 2D
    Tensor<int, 3> t3(2, 3, 4);  // tensor 3D
    Tensor<int, 4> t4(2, 2, 2, 2);  // tensor 4D

    // test 2: asignación de valores
    std::cout << std:: endl << "test 2: asignación de valores" << std::endl;
    t1 = {1, 2, 3, 4, 5};
    std::cout << "tensor 1D: " << std::endl << t1;

    // test 3: operaciones con escalares
    std::cout << std:: endl << "test 3: operaciones con escalares" << std::endl;
    auto t1_plus_2 = t1 + 2;
    std::cout << "t1 + 2: " << std::endl << t1_plus_2;
    auto t1_times_3 = t1 * 3;
    std::cout << "t1 * 3: " << std::endl << t1_times_3;

    // test 4: operaciones entre tensores
    std::cout << std:: endl << "test 4: operaciones entre tensores" << std::endl;
    Tensor<int, 2> t2_1(3, 4);
    Tensor<int, 2> t2_2(3, 4);
    t2_1 = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    t2_2 = {
        2, 3, 4, 5,
        6, 7, 8, 9,
        10, 11, 12, 13
    };
    auto t2_sum = t2_1 + t2_2;
    std::cout << "suma de tensores 2D: " << std::endl << t2_sum;

    // test 5: reshape
    std::cout << "test 5: reshape" << std::endl;
    t2_1.reshape(4, 3);
    std::cout << "tensor 2D reshaped a 4x3: " << std::endl << t2_1;

    // test 6: transpose
    std::cout << "test 6: transpose" << std::endl;
    auto t2_transposed = t2_1.transpose_2d();
    std::cout << "tensor 2D transpuesto: " << std::endl << t2_transposed;

    // test 7: matrix product
    std::cout << "test 7: matrix product" << std::endl;
    Tensor<int, 2> mat1(2, 3);
    Tensor<int, 2> mat2(3, 2);
    mat1 = {
        1, 2, 3,
        4, 5, 6
    };
    mat2 = {
        7, 8,
        9, 10,
        11, 12
    };
    auto mat_product = matrix_product(mat1, mat2);
    std::cout << "producto matricial: " << std::endl << mat_product;

    // test 8: broadcasting
    std::cout << "test 8: broadcasting" << std::endl;
    Tensor<int, 2> broad1(3, 1);
    Tensor<int, 2> broad2(1, 4);
    broad1 = {1, 2, 3};
    broad2 = {1, 2, 3, 4};
    auto broad_result = broad1 * broad2;
    std::cout << "resultado de broadcasting: " << std::endl << broad_result;

    // test 9: acceso a elementos
    std::cout << "test 9: acceso a elementos" << std::endl;
    std::cout << "elemento (1,1) del tensor: " << t2_1(1,1);

    // test 10: iteradores
    std::cout << std:: endl << std:: endl << "test 10: iteradores" << std::endl;
    std::cout << "elementos de t1 usando iteradores: ";
    for(auto it = t1.begin(); it != t1.end(); ++it) {
        std::cout << *it << " ";}
    std::cout << std::endl;

    return 0;}

#include <iostream>
#include "tensor.h"
using namespace std;

void test_case_1() {
    utec::algebra::Tensor<int, 2> t(2, 3);
    t.fill(7);
    int x = t(1, 2);
    cout << "caso 1: " << (x == 7 ? "éxito" : "fallo") << endl;}

void test_case_2() {
    utec::algebra::Tensor<int, 2> t2(2, 3);
    t2.reshape({3, 2});
    int y = t2[5];
    int y_check = t2(2, 1);
    cout << "caso 2: " << (y == y_check ? "éxito" : "fallo") << endl;}

void test_case_3() {
    bool exception_thrown = false;
    try {
        utec::algebra::Tensor<int, 3> t3(2, 2, 2);
        t3.reshape({2, 4, 1});}
    catch (const invalid_argument&) {
        exception_thrown = true;}
    cout << "caso 3: " << (exception_thrown ? "éxito" : "fallo") << endl;}

void test_case_4() {
    utec::algebra::Tensor<double, 2> a(2, 2), b(2, 2);
    a(0, 1) = 5.5;
    b.fill(2.0);
    auto sum = a + b;
    auto diff = sum - b;
    bool success = (sum(0, 1) == 7.5) && (diff(0, 1) == 5.5);
    cout << "caso 4: " << (success ? "éxito" : "fallo") << endl;}

void test_case_5() {
    utec::algebra::Tensor<float, 1> v(3);
    v.fill(2.0f);
    auto scaled = v * 4.0f;
    utec::algebra::Tensor<int, 3> cube(2, 2, 2);
    cube.fill(1);
    auto cube2 = cube * cube;
    bool success = (scaled(2) == 8.0f) && (cube2(1, 1, 1) == 1);
    cout << "caso 5: " << (success ? "éxito" : "fallo") << endl;}

void test_case_6() {
    utec::algebra::Tensor<int, 2> m(2, 1);
    m(0, 0) = 3; m(1, 0) = 4;
    utec::algebra::Tensor<int, 2> n(2, 3);
    n.fill(5);
    auto p = m * n;
    bool success = (p(0, 2) == 15) && (p(1, 1) == 20);
    cout << "caso 6: " << (success ? "éxito" : "fallo") << endl;}

void test_case_7() {
    utec::algebra::Tensor<int, 2> m2(2, 3);
    auto mt = m2.transpose_2d();
    bool success = (mt.shape() == array<size_t, 2>{3, 2}) && (mt(0, 1) == m2(1, 0));
    cout << "caso 7: " << (success ? "éxito" : "fallo");}

int main() {
    test_case_1();
    test_case_2();
    test_case_3();
    test_case_4();
    test_case_5();
    test_case_6();
    test_case_7();
    return 0;}

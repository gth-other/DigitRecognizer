#include "Tensor2D.hpp"


Tensor2D::Tensor2D() = default;
Tensor2D::Tensor2D(int32_t a, int32_t b) {
    this->a = a;
    this->b = b;

    this->data.resize(this->a * this->b);
}
float &Tensor2D::operator()(int32_t x, int32_t y) {
    return data[x * this->b + y];
}
float Tensor2D::operator()(int32_t x, int32_t y) const {
    return data[x * this->b + y];
}
float &Tensor2D::operator()(int32_t i) {
    return data[i];
}
float Tensor2D::operator()(int32_t i) const {
    return data[i];
}
int32_t Tensor2D::getA() const {
    return this->a;
}
int32_t Tensor2D::getB() const {
    return this->b;
}
int32_t Tensor2D::getSize() const {
    return (int32_t)this->data.size();
}
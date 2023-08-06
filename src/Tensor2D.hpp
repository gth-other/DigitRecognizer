#include <vector>
#include <cstdint>


#pragma once


class Tensor2D {
public:
    Tensor2D();
    Tensor2D(int32_t a, int32_t b);

    float &operator()(int32_t x, int32_t y);
    float operator()(int32_t x, int32_t y) const;

    float &operator()(int32_t i);
    float operator()(int32_t i) const;

    [[nodiscard]] int32_t getA() const;
    [[nodiscard]] int32_t getB() const;
    [[nodiscard]] int32_t getSize() const;
private:
    int32_t a, b;
    std::vector<float> data;
};
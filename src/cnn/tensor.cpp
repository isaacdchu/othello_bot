#include "tensor.h"

template <size_t X, size_t Y, size_t Z>
float Tensor<X, Y, Z>::at(size_t x, size_t y, size_t z) const {
    // Assumes valid indices are provided
    return data[z * X * Y + y * X + x];
}

template <size_t X, size_t Y, size_t Z>
void Tensor<X, Y, Z>::set(size_t x, size_t y, size_t z, float value) {
    // Assumes valid indices are provided
    data[z * X * Y + y * X + x] = value;
}

template <size_t X, size_t Y, size_t Z>
float Tensor<X, Y, Z>::dot(const Tensor<X, Y, Z> &other) const {
    // Assumes the dimensions are compatible for dot product
    float result = 0.0f;
    const auto& other_data = other.get_data();
    for (size_t i = 0; i < data.size(); i++) {
        result += data[i] * other_data[i];
    }
    return result;
}
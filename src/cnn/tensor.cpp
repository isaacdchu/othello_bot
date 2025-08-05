#include "tensor.h"

template <size_t X, size_t Y, size_t Z>
float Tensor<X, Y, Z>::at(size_t x, size_t y, size_t z) const {
    // Assumes valid indices are provided
    return data[x * Y * Z + y * Z + z];
}

template <size_t X, size_t Y, size_t Z>
void Tensor<X, Y, Z>::set(size_t x, size_t y, size_t z, float value) {
    // Assumes valid indices are provided
    data[x * Y * Z + y * Z + z] = value;
}

template <size_t X, size_t Y, size_t Z>
float Tensor<X, Y, Z>::dot(const Tensor<X, Y, Z> &other) const {
    // Assumes the dimensions are compatible for dot product
    float result = 0.0f;
    for (size_t i = 0; i < X; i++) {
        for (size_t j = 0; j < Y; j++) {
            for (size_t k = 0; k < Z; k++) {
                result += this->at(i, j, k) * other.at(i, j, k);
            }
        }
    }
    return result;
}
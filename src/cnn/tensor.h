#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <algorithm>

// Tensor class for representing a 3D tensor with fixed dimensions
// X, Y, Z are the dimensions of the tensor
// tensor.at(x, y, z) returns the value at position (x, y, z)
template <size_t X, size_t Y, size_t Z>
class Tensor {
private:
    std::array<float, X * Y * Z> data;
public:
    Tensor() {
        data.fill(0.0f); // Initialize all elements to zero
    }
    float at(size_t x, size_t y, size_t z) const;
    void set(size_t x, size_t y, size_t z, float value);
    const std::tuple<size_t, size_t, size_t>& shape() const { return std::make_tuple(X, Y, Z); };
    float dot(const Tensor<X, Y, Z> &other) const;
    const std::array<float, X * Y * Z>& get_data() const {
        return data; // Return the underlying data array
    }
};

#endif // TENSOR_H
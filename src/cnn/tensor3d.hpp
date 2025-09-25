#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "tensorinterface.hpp"

#include <array>

template<size_t X, size_t Y, size_t Z>
class Tensor3D : public TensorInterface {
private:
    std::array<float, X * Y * Z> data;
    std::array<float, X * Y * Z> result_data;
public:
    Tensor3D(const float init_value = 0.0f) {
        // Initialize the tensor with the given value
        data.fill(init_value);
        result_data.fill(0.0f);
    }

    Tensor3D(const std::array<float, X * Y * Z>& init_data) {
        // Initialize the tensor with provided data
        // Assumes init_data has the correct size
        data = init_data;
        result_data.fill(0.0f);
    }

    Tensor3D operator+(const Tensor3D& other) const {
        for (size_t i = 0; i < X * Y * Z; i++) {
            result_data[i] = data[i] + other.data[i];
        }
        return Tensor3D(result_data);
    }

    Tensor3D operator+(const float scalar) const {
        for (size_t i = 0; i < X * Y * Z; i++) {
            result_data[i] = data[i] + scalar;
        }
        return Tensor3D(result_data);
    }

    Tensor3D& operator+=(const Tensor3D& other) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    Tensor3D& operator+=(const float scalar) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            data[i] += scalar;
        }
        return *this;
    }

    Tensor3D operator-(const Tensor3D& other) const {
        for (size_t i = 0; i < X * Y * Z; i++) {
            result_data[i] = data[i] - other.data[i];
        }
        return Tensor3D(result_data);
    }

    Tensor3D operator-(const float scalar) const {
        for (size_t i = 0; i < X * Y * Z; i++) {
            result_data[i] = data[i] - scalar;
        }
        return Tensor3D(result_data);
    }

    Tensor3D& operator-=(const Tensor3D& other) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    Tensor3D& operator-=(const float scalar) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            data[i] -= scalar;
        }
        return *this;
    }

    const float& at(const size_t i, const size_t j, const size_t k) const {
        return data[k * X * Y + j * X + i];
    }

    float& at(const size_t i, const size_t j, const size_t k) {
        return data[k * X * Y + j * X + i];
    }

    const float& at(const size_t i) const {
        return data[i];
    }

    float& at(const size_t i) {
        return data[i];
    }
};

#endif // TENSOR_HPP
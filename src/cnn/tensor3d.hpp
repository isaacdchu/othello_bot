#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "tensorinterface.hpp"

#include <array>
#include <string>

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

    Tensor3D(const TensorInterface& other) {
        // Initialize from another TensorInterface
        for (size_t i = 0; i < X * Y * Z; i++) {
            data[i] = other.at(i);
        }
    }

    Tensor3D operator+(const Tensor3D& other) const {
        for (size_t i = 0; i < X * Y * Z; i++) {
            result_data[i] = data[i] + other.at(i);
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
            data[i] += other.at(i);
        }
        return *this;
    }

    Tensor3D& operator+=(const float scalar) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            data[i] += scalar;
        }
        return *this;
    }

    Tensor3D operator-(const Tensor3D& other) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            result_data[i] = data[i] - other.at(i);
        }
        return Tensor3D(result_data);
    }

    Tensor3D operator-(const float scalar) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            result_data[i] = data[i] - scalar;
        }
        return Tensor3D(result_data);
    }

    Tensor3D& operator-=(const Tensor3D& other) {
        for (size_t i = 0; i < X * Y * Z; i++) {
            data[i] -= other.at(i);
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

    void fill(float value) {
        data.fill(value);
    }

    std::string to_string() const {
        std::string result = "Tensor3D(";
        for (size_t k = 0; k < Z; k++) {
            result += "[";
            for (size_t j = 0; j < Y; j++) {
                result += "[";
                for (size_t i = 0; i < X; i++) {
                    result += std::to_string(at(i, j, k));
                    if (i < X - 1) result += ", ";
                }
                result += "]";
                if (j < Y - 1) result += ", ";
            }
            result += "]";
            if (k < Z - 1) result += ", ";
        }
        result += ")";
        return result;
    }

    constexpr size_t size() const {
        return X * Y * Z;
    }

    static constexpr std::array<size_t, 3> shape() {
        return {X, Y, Z};
    }
};

#endif // TENSOR_HPP
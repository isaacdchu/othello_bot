#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <algorithm>
#include <bitset>
#include <cstring>
#include <fstream>
#include <iostream>
#include <tuple>

// Tensor class for representing a 3D tensor with fixed dimensions
// X, Y, Z are the dimensions of the tensor
// tensor.at(x, y, z) returns the value at position (x, y, z)
template <size_t X, size_t Y, size_t Z>
class Tensor {
private:
    static constexpr size_t size = X * Y * Z;
    std::array<float, size> data;
public:
    Tensor() {
        // Initialize all elements to zero
        data.fill(0.0f); 
    }

    Tensor(const std::array<float, size>& init_data) : data(init_data) {
        // Constructor that initializes the tensor with provided data
        // Assumes init_data has the correct size (X * Y * Z) and indexing
    }

    Tensor(const std::string& line) {
        // Creates a tensor from a string representation
        std::string num_string = "";
        uint32_t bits = 0;
        float value = 0.0f;
        size_t i = 0;
        for (const char c : line) {
            if (c != ',') {
                num_string += c;
                continue;
            }
            bits = std::stoul(num_string);
            std::memcpy(&value, &bits, sizeof(float));
            data[i++] = value;
            num_string = "";
        }
        bits = std::stoul(num_string);
        std::memcpy(&value, &bits, sizeof(float));
        data[i] = value;
        num_string = "";
    }

    float at(size_t x, size_t y, size_t z) const {
        // Assumes valid indices are provided
        return data[z * X * Y + y * X + x];
    }

    float at(size_t i) const {
        // Assumes valid index is provided
        return data[i];
    }

    void set(size_t x, size_t y, size_t z, float value) {
        // Assumes valid indices are provided
        data[z * X * Y + y * X + x] = value;
    }

    void set(size_t i, float value) {
        // Assumes valid index is provided
        data[i] = value;
    }

    void clear() {
        // Sets all values in the tensor to zero
        data.fill(0.0f);
    }

    const std::tuple<size_t, size_t, size_t> shape() const { return std::make_tuple(X, Y, Z); };

    const std::array<float, size>& get_data() const {
        return data; // Return the underlying data array
    }

    std::string to_string() const {
        std::string result;
        for (size_t i = 0; i < size - 1; i++) {
            result = result + std::to_string(data[i]) + ",";
        }
        result += std::to_string(data[size - 1]);
        return result;
    }

    std::string save() const {
        // Represents the tensor as a string, with the second part compatible with the constructor
        // Return format: "number,number,...,number"
        std::string values;
        for (size_t i = 0; i < size - 1; i++) {
            uint32_t bits;
            std::memcpy(&bits, &data[i], sizeof(float));
            values += std::to_string(bits) + ",";
        }
        uint32_t bits;
        std::memcpy(&bits, &data[size - 1], sizeof(float));
        values += std::to_string(bits);
        return values;
    }

    float dot(const Tensor<X, Y, Z> &other) const {
        // Assumes the dimensions are compatible for dot product
        float result = 0.0f;
        const auto& other_data = other.get_data();
        for (size_t i = 0; i < data.size(); i++) {
            result += data[i] * other_data[i];
        }
        return result;
    }

    Tensor<X, Y, Z> operator *(const float scalar) const {
        // Scalar multiplication
        Tensor<X, Y, Z> result;
        for (size_t i = 0; i < size; i++) {
            result.set(i, data[i] * scalar);
        }
        return result;
    }

    Tensor<X, Y, Z> operator *(const Tensor<X, Y, Z> &other) const {
        // Element-wise multiplication
        Tensor<X, Y, Z> result;
        for (size_t i = 0; i < size; i++) {
            result.set(i, data[i] * other.at(i));
        }
        return result;
    }

    void operator *=(const float scalar) {
        // Element-wise multiplication
        for (size_t i = 0; i < size; i++) {
            data[i] *= scalar;
        }
    }

    Tensor<X, Y, Z> operator +(const Tensor<X, Y, Z> &other) const {
        // Element-wise addition
        Tensor<X, Y, Z> result;
        for (size_t i = 0; i < size; i++) {
            result.set(i, data[i] + other.at(i));
        }
        return result;
    }

    Tensor<X, Y, Z> operator +(const float scalar) const {
        // Element-wise addition
        Tensor<X, Y, Z> result;
        for (size_t i = 0; i < size; i++) {
            result.set(i, data[i] + scalar);
        }
        return result;
    }

    void operator +=(const Tensor<X, Y, Z> &other) {
        // Element-wise addition
        for (size_t i = 0; i < size; i++) {
            data[i] += other.data[i];
        }
    }

    Tensor<X, Y, Z> operator -(const Tensor<X, Y, Z> &other) const {
        // Element-wise subtraction
        Tensor<X, Y, Z> result;
        for (size_t i = 0; i < size; i++) {
            result.set(i, data[i] - other.data[i]);
        }
        return result;
    }

    void operator -=(const Tensor<X, Y, Z> &other) {
        // Element-wise subtraction
        for (size_t i = 0; i < size; i++) {
            data[i] -= other.data[i];
        }
    }

    Tensor<X, Y, Z> operator -(const float scalar) const {
        // Element-wise subtraction
        Tensor<X, Y, Z> result;
        for (size_t i = 0; i < size; i++) {
            result.set(i, data[i] - scalar);
        }
        return result;
    }
};

template <size_t X, size_t Y, size_t Z, size_t n>
Tensor<X, Y, Z*n> stack(const std::array<Tensor<X, Y, Z>, n>& tensors) {
    // Concatenates multiple tensors along the last dimension
    std::array<float, X * Y * Z * n> new_data;
    size_t index = 0;
    for (const auto& tensor : tensors) {
        const auto& tensor_data = tensor.get_data();
        std::copy(tensor_data.begin(), tensor_data.end(), new_data.begin() + index);
        index += tensor_data.size();
    }
    return Tensor<X, Y, Z*n>(new_data);
}

// Scuffed templating (n is only used for padding, not for array purposes)
template <size_t X, size_t Y, size_t Z, size_t n>
Tensor<X + 2*n, Y + 2*n, Z> pad(const std::array<Tensor<X, Y, Z>, n>& input) {
    // Zero-filled padding
    Tensor<X + 2*n, Y + 2*n, Z> padded_input = Tensor<X + 2*n, Y + 2*n, Z>();
    for (size_t z = 0; z < Z; z++) {
        for (size_t y = 0; y < Y; y++) {
            for (size_t x = 0; x < X; x++) {
                padded_input.set(x + n, y + n, z, input[0].at(x, y, z)); // Copy original values
            }
        }
    }
    return padded_input;
}

#endif // TENSOR_H
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

class Tensor {
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    std::vector<float> data_;
public:
    Tensor(const std::vector<size_t>& shape)
        : shape_(shape) {
        size_ = 1;
        for (size_t dim : shape) {
            size_ *= dim;
        }
        strides_.resize(shape.size());
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= shape[i];
        }
        data_.resize(size_, 0.0);
    }

    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
        : shape_(shape), data_(data) {
        size_ = 1;
        for (size_t dim : shape) {
            size_ *= dim;
        }
        if (data.size() != size_) {
            throw std::invalid_argument("Data size does not match tensor shape.");
        }
        strides_.resize(shape.size());
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= shape[i];
        }
    }
    
    Tensor(const Tensor& other)
        : shape_(other.shape_), size_(other.size_), data_(other.data_), strides_(other.strides_) {
    }

    Tensor operator+(const Tensor& other) const {
        Tensor result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor operator+(float scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] + scalar;
        }
        return result;
    }

    Tensor& operator+=(const Tensor& other) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Tensor& operator+=(float scalar) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] += scalar;
        }
        return *this;
    }

    Tensor operator-(const Tensor& other) const {
        Tensor result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Tensor operator-(float scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] - scalar;
        }
        return result;
    }

    Tensor& operator-=(const Tensor& other) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Tensor& operator-=(float scalar) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] -= scalar;
        }
        return *this;
    }

    Tensor operator*(float scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    Tensor& operator*=(float scalar) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    bool operator==(const Tensor& other) const {
        return shape_ == other.shape_ && data_ == other.data_;
    }
    
    void reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != size_) {
            throw std::invalid_argument("New shape must have the same total size as the old shape.");
        }

        strides_.resize(new_shape.size());
        size_t stride = 1;
        for (size_t i = new_shape.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= new_shape[i];
        }
        
        shape_ = new_shape;
    }

    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    std::string to_string() const {
        std::string result = "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            result += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) {
                result += ", ";
            }
        }
        result += "], data=[";
        for (size_t i = 0; i < data_.size(); ++i) {
            result += std::to_string(data_[i]);
            if (i < data_.size() - 1) {
                result += ", ";
            }
        }
        result += "])";
        return result;
    }

    std::string serialize() const {
        std::ostringstream data_stream;
        data_stream.precision(10); // Set precision to 10 digits
        data_stream << std::fixed; // Use fixed-point notation
        for (size_t i = 0; i < size_; i++) {
            data_stream << data_[i]  << ",";
        }
        data_stream << "\n";
        for (size_t dim : shape_) {
            data_stream << dim << ",";
        }
        data_stream << "\n";
        std::string serialized_data = data_stream.str();
        return serialized_data;
    }

    const float& at(const std::vector<size_t>& indices) const {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides_[i];
        }
        return data_[index];
    }

    float& at(const std::vector<size_t>& indices) {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides_[i];
        }
        return data_[index];
    }

    template <typename... Indices>
    const float& at(Indices... indices) const {
        std::vector<size_t> idx = {static_cast<size_t>(indices)...};
        return at(idx);
    }

    template <typename... Indices>
    float& at(Indices... indices) {
        std::vector<size_t> idx = {static_cast<size_t>(indices)...};
        return at(idx);
    }

    const std::vector<size_t>& shape() const {
        return shape_;
    }

    constexpr size_t size() const {
        return size_;
    }

    const std::vector<float>& data() const {
        return data_;
    }
};

#endif // TENSOR_HPP
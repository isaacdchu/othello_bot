#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "tensor.hpp"

#include <cstddef>
#include <vector>
#include <functional>
#include <stdexcept>

class DenseLayer : public Layer {
private:
    const std::vector<std::size_t> input_shape_;
    const std::vector<std::size_t> output_shape_;
    Tensor weights_;
    Tensor biases_;
    bool calculate_gradients_;
public:
    DenseLayer(const std::vector<std::size_t>& input_shape, const std::vector<std::size_t>& output_shape) : 
    input_shape_(input_shape),
    output_shape_(output_shape),
    weights_(std::vector<std::size_t>{input_shape.back(), output_shape.back()}), 
    biases_(std::vector<std::size_t>{output_shape.back()}),
    calculate_gradients_(true) {
        if (input_shape.size() != 1 || output_shape.size() != 1) {
            throw std::invalid_argument("DenseLayer only supports 1D input and output shapes.");
        }
    }

    void train() override {
        calculate_gradients_ = true;
    }

    void eval() override {
        calculate_gradients_ = false;
    }

    Tensor forward(const Tensor& input) override {
        Tensor result = Tensor(biases_);
        for (std::size_t i = 0; i < output_shape_.back(); i++) {
            float sum = 0.0f;
            for (std::size_t j = 0; j < input_shape_.back(); j++) {
                sum += input.at(j) * weights_.at(j, i);
            }
            result.at(i) += sum;
        }
        return result;
    }

    std::vector<std::reference_wrapper<Tensor>> parameters() override {
        return {std::ref(weights_), std::ref(biases_)};
    }
};

#endif // DENSE_LAYER_HPP
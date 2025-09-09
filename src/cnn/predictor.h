#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "tensor.h"
#include <string>
#include <array>
#include <fstream>

struct CNNOutput {
    uint64_t best_move;
    float board_evaluation;
};

class Predictor {
public:
    Predictor(const std::string& model_path) {
        // Load the model from the specified path
        std::string line;
        std::ifstream model_file(model_path);
        if (!model_file) {
            throw std::runtime_error("(CNN) Failed to open model file: " + model_path);
        }
        // Load all model parameters:
        for (auto &filter : filter_weights) {
            std::getline(model_file, line);
            filter = Tensor<3, 3, 3>(line);
        }
        for (float &bias : filter_biases) {
            std::getline(model_file, line);
            bias = std::stof(line);
        }
        for (auto &weights : flatten_weights) {
            for (float &weight : weights) {
                std::getline(model_file, line);
                weight = std::stof(line);
            }
        }
        for (float &bias : flatten_biases) {
            std::getline(model_file, line);
            bias = std::stof(line);
        }
        for (float &weight : policy_weights) {
            std::getline(model_file, line);
            weight = std::stof(line);
        }
        for (float &bias : policy_biases) {
            std::getline(model_file, line);
            bias = std::stof(line);
        }
        for (float &weight : value_weights) {
            std::getline(model_file, line);
            weight = std::stof(line);
        }
        std::getline(model_file, line);
        value_bias = std::stof(line);
    }
    CNNOutput predict(const Tensor<8, 8, 3>& input) const {
        std::array<Tensor<8, 8, 1>, 32> conv_results = {};
        for (size_t i = 0; i < filter_weights.size(); ++i) {
            conv_results[i] = convolution(input, filter_weights[i], filter_biases[i]);
        }
        Tensor<8, 8, 32> stacked_result = stack(conv_results); // Concatenate results along the last dimension
        Tensor<8, 8, 2> flattened_result = flatten(stacked_result); // Flatten the result to 8x8x2
        Tensor<8, 8, 1> policy_output = dense_policy(flattened_result);
        float value_output = dense_value(flattened_result);
        uint64_t best_move = 0;
        std::array<float, 64> policy_array = policy_output.get_data();
        for (uint64_t i = 0; i < 64; i++) {
            if (policy_array[i] > policy_array[best_move]) {
                best_move = i;
            }
        }
        return {best_move, value_output};
    }
private:
    const float epsilon = 1e-8f; // Small value to avoid division by zero

    // Model parameters
    std::array<Tensor<3, 3, 3>, 32> filter_weights; // 32 filters of size 3x3
    std::array<float, 32> filter_biases; // Biases for each filter
    std::array<std::array<float, 32>, 2> flatten_weights; // Two 1x1x32 tensors for flattening
    std::array<float, 2> flatten_biases; // Biases for flattening to 64 outputs
    std::array<float, 128 * 64> policy_weights; // Weights for the dense layer from flattened output to policy head
    std::array<float, 64> policy_biases; // Biases for the policy head
    std::array<float, 128> value_weights; // Weights for the dense layer from flattened output to value head
    float value_bias; // Bias for the value head

    Tensor<8, 8, 1> convolution(const Tensor<8, 8, 3> &input, const Tensor<3, 3, 3> &filter, const float bias) const {
        const std::array<Tensor<8, 8, 3>, 1> temp_input = {input};
        Tensor<10, 10, 3> padded_input = pad(temp_input);
        Tensor<8, 8, 1> output_tensor = Tensor<8, 8, 1>();
        for (size_t y = 1; y <= 8; y++) {
            for (size_t x = 1; x <= 8; x++) {
                Tensor<3, 3, 3> region = Tensor<3, 3, 3>();
                for (size_t z = 0; z < 3; z++) {
                    for (size_t j = 0; j < 3; j++) {
                        for (size_t i = 0; i < 3; i++) {
                            region.set(i, j, z, padded_input.at(x - 1 + i, y - 1 + j, z));
                        }
                    }
                }
                // Perform dot product
                const float output_value = activation_function(region.dot(filter) + bias);
                output_tensor.set(x - 1, y - 1, 0, output_value);
            }
        }
        return output_tensor;
    }
    Tensor<8, 8, 2> flatten(const Tensor<8, 8, 32> &input) const {
        Tensor<8, 8, 2> output_tensor = Tensor<8, 8, 2>();
        for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
                float sum_0 = flatten_biases[0];
                float sum_1 = flatten_biases[1];
                for (size_t z = 0; z < 32; z++) {
                    sum_0 += input.at(x, y, z) * flatten_weights[0][z];
                    sum_1 += input.at(x, y, z) * flatten_weights[1][z];
                }
                output_tensor.set(x, y, 0, sum_0);
                output_tensor.set(x, y, 1, sum_1);
            }
        }
        return output_tensor;
    }
    Tensor<8, 8, 1> dense_policy(const Tensor<8, 8, 2> &input) const {
        std::array<float, 64> output_array = {};
        for (size_t i = 0; i < 64; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < 128; j++) {
                const float p_w = policy_weights[i * 128 + j];
                sum += input.at(j) * p_w;
            }
            sum += policy_biases[i];
            output_array[i] = sum;
        }
        const std::array<float, 64> softmaxed_output_array = softmax(output_array);
        return Tensor<8, 8, 1>(softmaxed_output_array);
    }
    float dense_value(const Tensor<8, 8, 2> &input) const {
        float sum = value_bias; // Start with the bias
        for (size_t i = 0; i < 128; i++) {
            sum += input.at(i) * value_weights[i];
        }
        return std::tanh(sum);
    }
    float activation_function(float x) const {
        // ReLU activation function
        return x > 0 ? x : 0;
    }
    std::array<float, 64> softmax(const std::array<float, 64> &x) const {
        std::array<float, 64> result = {};
        const float max_val = *std::max_element(x.begin(), x.end());
        float sum = epsilon;
        for (const float val : x) {
            sum += std::exp(val - max_val);
        }
        for (size_t i = 0; i < 64; i++) {
            result[i] = std::exp(x[i] - max_val) / sum;
            if (result[i] != result[i]) {
                throw std::runtime_error("softmax result is nan");
            }
            if (std::isinf(result[i])) {
                throw std::runtime_error("softmax result is inf");
            }
        }
        return result;
    }
};

#endif // PREDICTOR_H
#ifndef CNN_H
#define CNN_H

#include "../board/board.h"
#include "../mcts/mctsnode.h"
#include "tensor.h"

#include <string>
#include <iostream>
#include <fstream>
#include <array>
#include <tuple>
#include <cmath>

struct Output {
    Tensor<8, 8, 1> policy; // Policy vector for best moves
    float value; // Evaluation of the board state for the current player (black)
};

class CNN {
private:
    // Used for Adam optimizer
    const float learning_rate;
    unsigned int adam_t; // Time step

public:
    CNN(const float learning_rate = 0.001f) : learning_rate(learning_rate), adam_t(0) {
        // Initialize the CNN with small random weights and biases
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.001f, 0.01f); // Small random values for weights
        for (auto &filter : filter_weights) {
            for (size_t z = 0; z < 3; z++) {
                for (size_t y = 0; y < 3; y++) {
                    for (size_t x = 0; x < 3; x++) {
                        filter.set(x, y, z, dist(gen));
                    }
                }
            }
        }
        for (auto &bias : filter_biases) {
            bias = dist(gen); // Initialize biases to small random values
        }
        for (auto &flatten_weight : flatten_weights) {
            for (size_t z = 0; z < 32; z++) {
                flatten_weight[z] = dist(gen);
            }
        }
        for (auto &bias : flatten_biases) {
            bias = dist(gen); // Initialize biases to small random values
        }
        for (size_t i = 0; i < 128 * 64; i++) {
            policy_weights[i] = dist(gen); // Initialize policy weights to small random values
        }
        for (auto &bias : policy_biases) {
            bias = dist(gen); // Initialize policy biases to small random values
        }
        for (size_t i = 0; i < 128; i++) {
            value_weights[i] = dist(gen); // Initialize value weights to small random values
        }
        value_bias = dist(gen); // Initialize value bias to a small random value
    };
    CNN(const std::string &model_path, const float learning_rate = 0.001f) : learning_rate(learning_rate), adam_t(0) {
        // TODO
        // Load the model from the specified path
    }
    void train(const std::string &data_path);
    Output predict(const std::string &data_path) const;
    Output predict(const Tensor<8, 8, 3> &input) const;
    void save_model(const std::string &model_path) const;
    static std::pair<Tensor<8, 8, 3>, const float> parse_line(const std::string &line);

private:
    // Used for Adam optimizer
    const float beta_1 = 0.9;
    const float beta_2 = 0.999;
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

    // Saved intermediate steps
    Tensor<8, 8, 32> stacked_result;
    Tensor<8, 8, 2> flattened_results;
    Tensor<8, 8, 1> policy_results;
    float value_result;

    Tensor<8, 8, 1> convolution(const Tensor<8, 8, 3> &input, const Tensor<3, 3, 3> &filter, const float bias) const;
    Tensor<8, 8, 2> flatten(const Tensor<8, 8, 32> &input) const;
    Tensor<8, 8, 1> dense_policy(const Tensor<8, 8, 2> &input) const;
    float dense_value(const Tensor<8, 8, 2> &input) const;
    float activation_function(float x) const {
        // ReLU activation function
        return x > 0 ? x : 0;
    }
    std::array<float, 64> softmax(const std::array<float, 64> &x) const {
        // o(z) = exp(z) / sum(exp(z_i)) for all i
        std::array<float, 64> result = {};
        float sum = epsilon;
        for (const auto &val : x) {
            // Calculate the denominator
            sum += std::exp(val);
        }
        for (size_t i = 0; i < 64; i++) {
            result[i] = std::exp(x[i]) / sum;
        }
        return result;
    }
    float policy_loss(const std::array<float, 64> &predicted, const std::array<float, 64> &target) const {
        // Cross-entropy loss for policy head: - sum(target * log(predicted))
        // Values are move probabilities, so they should sum to 1
        float loss = 0.0f;
        for (size_t i = 0; i < 64; i++) {
            loss -= target[i] * std::log(predicted[i] + epsilon);
        }
        return loss;
    }
    std::array<float, 64> policy_loss_gradients(const std::array<float, 64> &predicted, const std::array<float, 64> &target) const {
        // Gradient of the policy loss with respect to the predicted probabilities
        std::array<float, 64> gradients = {};
        for (size_t i = 0; i < 64; i++) {
            gradients[i] = predicted[i] - target[i]; // dL/dp = p - t
        }
        return gradients;
    }
    float value_loss(float predicted, float target) const {
        // Mean squared error loss for value head: 0.5 * (predicted - target)^2
        // Values are win probabilities, so they should be in the range [-1, 1]
        float diff = predicted - target;
        return 0.5f * diff * diff;
    }
    float value_loss_gradient(float predicted, float target) const {
        // Gradient of the value loss with respect to the predicted value
        return predicted - target; // dL/dv = v - t
    }
    Output forward_pass(const Tensor<8, 8, 3> &input_tensor);
    void backward_pass(const Output &output, const Output &label);
};

#endif // CNN_H
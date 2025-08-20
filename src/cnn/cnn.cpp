#include "cnn.h"

void CNN::train(const std::string &data_path) {
    std::string line;
    std::ifstream data(data_path); // Open the training data file
    if (!data) {
        throw std::runtime_error("Failed to open data file: " + data_path);
    }
    while (std::getline(data, line)) {
        auto [input_tensor, value] = parse_line(line); // Parse the input tensor and value
        // Create the target output
        Board board = Board(input_tensor);
        MCTSNode root = MCTSNode(0, board, nullptr, true); // Black is the root player
        root.initialize_children();
        const unsigned int max_iterations = 10000;
        const unsigned int num_simulations = 10;
        for (unsigned int i = 0; i < max_iterations; i++) {
            auto node = root.select();
            if (node == nullptr) break;
            for (unsigned int j = 0; j < num_simulations; j++) {
                const float result = node->simulate();
                node->backpropagate(result);
            }
        }
        std::array<float, 64> policy = root.get_policy(); // Get the policy distribution from MCTS
        Output target = {Tensor<8, 8, 1>(policy), static_cast<float>(value)};
        Output results = forward_pass(input_tensor); // Get the CNN's prediction
        backward_pass(results, target); // Update the CNN based on the prediction and label
        float loss = policy_loss(results.policy.get_data(), target.policy.get_data()); // Calculate the policy loss
        loss += value_loss(results.value, target.value); // Add the value loss
    }
}

Output CNN::predict(const std::string &data_path) const { 
    // Placeholder for predicting from a file
    return {
        Tensor<8, 8, 1>(),
        0.0f
    };
}

Output CNN::predict(const Tensor<8, 8, 3> &input) const {
    // Perform convolution with each filter and accumulate results
    std::array<Tensor<8, 8, 1>, 32> conv_results = {};
    for (size_t i = 0; i < filter_weights.size(); ++i) {
        conv_results[i] = convolution(input, filter_weights[i], filter_biases[i]);
    }
    Tensor<8, 8, 32> stacked_result = stack(conv_results); // Concatenate results along the last dimension
    Tensor<8, 8, 2> flattened_result = flatten(stacked_result); // Flatten the result to 8x8x2
    Tensor<8, 8, 1> policy_output = dense_policy(flattened_result);
    float value_output = dense_value(flattened_result);
    return {policy_output, value_output}; // Return the output structure
}

void CNN::save_model(const std::string &model_path) const {
    // TODO: Implement saving the model to a file
}

Tensor<8, 8, 1> CNN::convolution(const Tensor<8, 8, 3> &input, const Tensor<3, 3, 3> &filter, const float bias) const {
    // Perform convolution operation on the input tensor with the given filter
    // Uses "same" padding and stride of 1
    Tensor<10, 10, 3> padded_input = Tensor<10, 10, 3>();
    for (size_t z = 0; z < 3; z++) {
        for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
                padded_input.set(x + 1, y + 1, z, input.at(x, y, z));
                if (x == 0) {
                    padded_input.set(0, y + 1, z, input.at(0, y, z)); // Left padding
                } else if (x == 7) {
                    padded_input.set(9, y + 1, z, input.at(7, y, z)); // Right padding
                }
                if (y == 0) {
                    padded_input.set(x + 1, 0, z, input.at(x, 0, z)); // Top padding
                } else if (y == 7) {
                    padded_input.set(x + 1, 9, z, input.at(x, 7, z)); // Bottom padding
                }
            }
        }
    }
    // Perform convolution operation
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
            output_tensor.set(x - 1, y - 1, 0, activation_function(region.dot(filter) + bias));
        }
    }
    return output_tensor;
}

Tensor<8, 8, 2> CNN::flatten(const Tensor<8, 8, 32> &input) const {
    // Flatten the input tensor to a 2D tensor of size 8x8x2 using two 1x1x32 convolutional layers
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

Tensor<8, 8, 1> CNN::dense_policy(const Tensor<8, 8, 2> &input) const {
    std::array<float, 64> output_array = {};
    for (size_t i = 0; i < 64; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < 128; j++) {
            sum += input.at(j) * policy_weights[i * 128 + j];
        }
        sum += policy_biases[i];
        output_array[i] = sum;
    }
    return Tensor<8, 8, 1>(softmax(output_array));
}

float CNN::dense_value(const Tensor<8, 8, 2> &input) const {
    // Dense layer to value head
    float sum = value_bias; // Start with the bias
    for (size_t i = 0; i < 128; i++) {
        sum += input.at(i) * value_weights[i];
    }
    return std::tanh(sum);
}

Output CNN::forward_pass(const Tensor<8, 8, 3> &input_tensor) {
    // Perform convolution with each filter and accumulate results
    std::array<Tensor<8, 8, 1>, 32> conv_results = {};
    for (size_t i = 0; i < filter_weights.size(); i++) {
        conv_results[i] = convolution(input_tensor, filter_weights[i], filter_biases[i]);
    }
    stacked_result = stack(conv_results); // Concatenate results along the last dimension
    flattened_results = flatten(stacked_result); // Flatten the result to 8x8x2 with two 1x1 convolutions
    policy_results = dense_policy(flattened_results);
    value_result = dense_value(flattened_results);
    Output output = {policy_results, value_result};
    return output;
}

void CNN::backward_pass(const Output &output, const Output &label) {
    // Using Adam optimizer for backpropagation
    // Policy gradients
    std::array<float, 64> policy_dL_dZ = {};
    for (size_t i = 0; i < 64; i++) {
        policy_dL_dZ[i] = output.policy.at(i) - label.policy.at(i);
    }
    std::array<float, 128 * 64> policy_dense_weight_dL_dW = {};
    for (size_t i = 0; i < 128 * 64; i++) {
        policy_dense_weight_dL_dW[i] = policy_dL_dZ[i % 64] * flattened_results.at(i / 64);
    }
    std::array<float, 64> policy_dense_bias_dL_dB = {};
    for (size_t i = 0; i < 64; i++) {
        policy_dense_bias_dL_dB[i] = policy_dL_dZ[i];
    }
    std::array<float, 128> policy_dense_dL_dY = {};
    for (size_t i = 0; i < 128; i++) {
        for (size_t j = 0; j < 64; j++) {
            policy_dense_dL_dY[i] += policy_dL_dZ[j] * policy_weights[i * 64 + j];
        }
    }

    // Value gradients
    float value_dL_dZ = (output.value - label.value) * (1.0f - std::tanh(output.value) * std::tanh(output.value));
    std::array<float, 128> value_dense_weight_dL_dW = {};
    for (size_t i = 0; i < 128; i++) {
        value_dense_weight_dL_dW[i] = value_dL_dZ * flattened_results.at(i);
    }
    float value_dense_bias_dL_dB = value_dL_dZ;
    std::array<float, 128> value_dense_dL_dY = {};
    for (size_t i = 0; i < 128; i++) {
        value_dense_dL_dY[i] += value_dL_dZ * value_weights[i];
    }

    std::array<float, 128> total_dense_dL_dY = {};
    for (size_t i = 0; i < 128; i++) {
        total_dense_dL_dY[i] = policy_dense_dL_dY[i] + value_dense_dL_dY[i];
    }

    // Flattening gradients
    Tensor<8, 8, 2> flatten_dL_dZ = Tensor<8, 8, 2>();
    for (size_t i = 0; i < 128; i++) {
        // No activation function for flattening (activation is f(x) = x, so derivative = 1)
        flatten_dL_dZ.set(i, total_dense_dL_dY[i]);
    }
    std::array<std::array<float, 32>, 2> flatten_weights_dL_dW = {};
    std::array<float, 2> flatten_bias_dL_dB = {};
    for (size_t i = 0; i < 32; i++) {
        for (size_t j = 0; j < 64; j++) {
            flatten_weights_dL_dW[0][i] += flatten_dL_dZ.at(j) * stacked_result.at(i * 64 + j);
            flatten_bias_dL_dB[0] += flatten_dL_dZ.at(j);
        }
        for (size_t j = 64; j < 128; j++) {
            flatten_weights_dL_dW[1][i] += flatten_dL_dZ.at(j) * stacked_result.at((i - 1) * 64 + j);
            flatten_bias_dL_dB[1] += flatten_dL_dZ.at(j);
        }
    }
    Tensor<8, 8, 32> flatten_dL_dY = Tensor<8, 8, 32>();
    for (size_t z = 0; z < 32; z++) {
        for (size_t y = 0; y < 8; y++) {
            for (size_t x = 0; x < 8; x++) {
                flatten_dL_dY.set(x, y, z, flatten_dL_dZ.at(x, y, 0) * flatten_weights_dL_dW[0][z]);
                flatten_dL_dY.set(x, y, z, flatten_dL_dZ.at(x, y, 1) * flatten_weights_dL_dW[1][z]);
            }
        }
    }

    // Filter gradients
    Tensor<8, 8, 32> filter_dL_dZ = Tensor<8, 8, 32>();
    for (size_t i = 0; i < 8 * 8 * 32; i++) {
        float f_prime_Z = stacked_result.at(i) > 0 ? 1.0f : 0.0f; // ReLU derivative
        filter_dL_dZ.set(i, flatten_dL_dY.at(i) * f_prime_Z);
    }
    std::array<Tensor<3, 3, 3>, 32> filter_dL_dW = {};
    std::array<float, 32> filter_dL_dB = {};
    // TODO: finish calculating gradients for filter weights and biases

    // Update time step before next iteration
    adam_t++;
}

std::pair<Tensor<8, 8, 3>, const float> CNN::parse_line(const std::string &line) {
    // Assumes "line" is properly formatted (indices from 0 to 384 inclusive)
    // Parses a line from the training data file
    std::array<float, 8 * 8 * 3> init_data;
    float evaluation = std::tanh(std::stof(line.substr(384))); // Extract evaluation label from the line
    for (size_t i = 0; i < 383; i += 2) {
        init_data[i / 2] = line[i] - '0'; // Convert character to float
    }
    Tensor<8, 8, 3> tensor = Tensor<8, 8, 3>(init_data);
    return {tensor, evaluation};
}
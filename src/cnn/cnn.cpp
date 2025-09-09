#include "cnn.h"

void CNN::train(const std::string &data_path, const unsigned int num_lines) {
    // Set num_lines to 0 for all lines
    std::ifstream data(data_path); // Open the training data file
    if (!data) {
        throw std::runtime_error("(train) Failed to open data file: " + data_path);
    }
    std::string line;
    unsigned int line_count = 0;
    while (std::getline(data, line)) {
        if (num_lines != 0 && line_count >= num_lines) break;
        std::cout << "Processing line " << ++line_count << std::endl;
        auto data = parse_line(line); // Parse the input tensor and value
        input_tensor = data.first;
        const float value = data.second;

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
        // float loss = policy_loss(results.policy.get_data(), target.policy.get_data()); // Calculate the policy loss
        // loss += value_loss(results.value, target.value); // Add the value loss
    }
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

Metrics CNN::evaluate(const std::string &data_path, const unsigned int num_lines) const {
    std::ifstream data(data_path);
    if (!data) {
        throw std::runtime_error("(evaluate) Failed to open data file: " + data_path);
    }
    std::string line;
    unsigned int line_count = 0;
    Tensor<8, 8, 3> temp;
    Tensor<8, 8, 3> &input_tensor = temp;
    Metrics metrics = {0.0f, 0.0f};
    while (std::getline(data, line)) {
        if (num_lines != 0 && line_count >= num_lines) break;
        std::cout << "Evaluating line " << ++line_count << std::endl;
        auto data = parse_line(line); // Parse the input tensor and value
        input_tensor = data.first;
        Output results = predict(input_tensor); // Get the CNN's prediction

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
        const std::array<float, 64> &policy = root.get_policy(); // Get the policy distribution from MCTS
        const float value = data.second; // Evaluation label
        // Compute and accumulate metrics
        metrics.policy_loss += policy_loss(results.policy.get_data(), policy);
        metrics.value_loss += value_loss(results.value, value);
    }
    if (line_count > 0) {
        metrics.policy_loss /= line_count;
        metrics.value_loss /= line_count;
    }
    return metrics;
}

void CNN::save_model(const std::string &model_path) const {
    std::ofstream model_file(model_path);
    if (!model_file) {
        throw std::runtime_error("(save_model) Failed to open model file: " + model_path);
    }
    // Saves all model parameters:
    // std::array<Tensor<3, 3, 3>, 32> filter_weights;       // 32 filters of size 3x3
    for (const auto& filter : filter_weights) {
        model_file << filter.save() << "\n";
    }
    // std::array<float, 32> filter_biases;                  // Biases for each filter
    for (const float& bias : filter_biases) {
        model_file << bias << "\n";
    }
    // std::array<std::array<float, 32>, 2> flatten_weights; // Two 1x1x32 tensors for flattening
    for (const auto& weights : flatten_weights) {
        for (const float weight : weights) {
            model_file << weight << "\n";
        }
    }
    // std::array<float, 2> flatten_biases;                  // Biases for flattening to 64 outputs
    for (const float bias : flatten_biases) {
        model_file << bias << "\n";
    }
    // std::array<float, 128 * 64> policy_weights;           // Weights for the dense layer from flattened output to policy head
    for (const float weight : policy_weights) {
        model_file << weight << "\n";
    }
    // std::array<float, 64> policy_biases;                  // Biases for the policy head
    for (const float bias : policy_biases) {
        model_file << bias << "\n";
    }
    // std::array<float, 128> value_weights;                 // Weights for the dense layer from flattened output to value head
    for (const float weight : value_weights) {
        model_file << weight << "\n";
    }
    // float value_bias;                                     // Bias for the value head
    model_file << value_bias << std::endl;
}

Tensor<8, 8, 1> CNN::convolution(const Tensor<8, 8, 3> &input, const Tensor<3, 3, 3> &filter, const float bias) const {
    // Perform convolution operation on the input tensor with the given filter
    // Uses zero-filled padding and stride of 1
    const std::array<Tensor<8, 8, 3>, 1> temp_input = {input};
    Tensor<10, 10, 3> padded_input = pad(temp_input);

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
            const float output_value = activation_function(region.dot(filter) + bias);
            output_tensor.set(x - 1, y - 1, 0, output_value);
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
            const float p_w = policy_weights[i * 128 + j];
            sum += input.at(j) * p_w;
        }
        sum += policy_biases[i];
        output_array[i] = sum;
    }
    const std::array<float, 64> softmaxed_output_array = softmax(output_array);
    return Tensor<8, 8, 1>(softmaxed_output_array);
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
    // Calculates necessary gradients for updating parameters
    // Policy gradients
    std::array<float, 64> policy_dL_dZ = {};
    for (size_t i = 0; i < 64; i++) {
        policy_dL_dZ[i] = output.policy.at(i) - label.policy.at(i);
    }
    std::array<float, 128 * 64> policy_dense_weight_dL_dW = {};
    for (size_t i = 0; i < 128 * 64; i++) {
        const float p_dL_dZ = policy_dL_dZ[i % 64];
        const float f_result = flattened_results.at(i / 64);
        policy_dense_weight_dL_dW[i] = p_dL_dZ * f_result;
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
    std::array<float, 2> flatten_biases_dL_dB = {};
    for (size_t i = 0; i < 32; i++) {
        for (size_t j = 0; j < 64; j++) {
            flatten_weights_dL_dW[0][i] += flatten_dL_dZ.at(j) * stacked_result.at(i * 64 + j);
            flatten_biases_dL_dB[0] += flatten_dL_dZ.at(j);
        }
        for (size_t j = 64; j < 128; j++) {
            flatten_weights_dL_dW[1][i] += flatten_dL_dZ.at(j) * stacked_result.at((i - 1) * 64 + j);
            flatten_biases_dL_dB[1] += flatten_dL_dZ.at(j);
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
    std::array<Tensor<3, 3, 3>, 32> filter_weights_dL_dW = {};
    const std::array<Tensor<8, 8, 3>, 1> temp_input = {input_tensor};
    const Tensor<10, 10, 3> padded_input_tensor = pad(temp_input);
    // dL/dW(k) = X star dL/dZ(k)
    for (size_t k = 0; k < 32; k++) {
        for (size_t c = 0; c < 3; c++) {
            for (size_t v = 0; v < 3; v++) {
                for (size_t u = 0; u < 3; u++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < 8; j++) {
                        for (size_t i = 0; i < 8; i++) {
                            sum += padded_input_tensor.at(i + u, j + v, c) * filter_dL_dZ.at(i, j, k);
                        }
                    }
                    filter_weights_dL_dW[k].set(u, v, c, sum);
                }
            }
        }
    }
    std::array<float, 32> filter_biases_dL_dB = {};
    for (size_t i = 0; i < 32; i++) {
        for (size_t j = 0; j < 64; j++) {
            filter_biases_dL_dB[i] += filter_dL_dZ.at(i * 64 + j);
        }
    }

    // Update parameters using Adam optimizer
    // Policy weights
    for (size_t i = 0; i < policy_dense_weight_dL_dW.size(); i++) {
        const float gradient = policy_dense_weight_dL_dW[i];
        const float update = get_update(gradient, policy_weights_m_t_prev[i], policy_weights_v_t_prev[i], beta_1, beta_2, learning_rate, epsilon, adam_t);
        // Apply updates
        policy_weights[i] -= update;
    }

    // Policy biases
    for (size_t i = 0; i < policy_dense_bias_dL_dB.size(); i++) {
        const float gradient = policy_dense_bias_dL_dB[i];
        const float update = get_update(gradient, policy_biases_m_t_prev[i], policy_biases_v_t_prev[i], beta_1, beta_2, learning_rate, epsilon, adam_t);
        // Apply updates
        policy_biases[i] -= update;
    }

    // Value weights
    for (size_t i = 0; i < value_dense_weight_dL_dW.size(); i++) {
        const float gradient = value_dense_weight_dL_dW[i];
        const float update = get_update(gradient, value_weights_m_t_prev[i], value_weights_v_t_prev[i], beta_1, beta_2, learning_rate, epsilon, adam_t);
        // Apply updates
        value_weights[i] -= update;
    }

    // Value bias
    const float gradient = value_dense_bias_dL_dB;
    const float update = get_update(gradient, value_bias_m_t_prev, value_bias_v_t_prev, beta_1, beta_2, learning_rate, epsilon, adam_t);
    value_bias -= update;

    // Flatten weights
    for (size_t i = 0; i < flatten_weights_dL_dW.size(); i++) {
        for (size_t j = 0; j < flatten_weights_dL_dW[i].size(); j++) {
            const float gradient = flatten_weights_dL_dW[i][j];
            const float update = get_update(gradient, flatten_weights_m_t_prev[i][j], flatten_weights_v_t_prev[i][j], beta_1, beta_2, learning_rate, epsilon, adam_t);
            // Apply updates
            flatten_weights[i][j] -= update;
        }
    }

    // Flatten biases
    for (size_t i = 0; i < flatten_biases_dL_dB.size(); i++) {
        const float gradient = flatten_biases_dL_dB[i];
        const float update = get_update(gradient, flatten_biases_m_t_prev[i], flatten_biases_v_t_prev[i], beta_1, beta_2, learning_rate, epsilon, adam_t);
        flatten_biases[i] -= update;
    }

    // Filter weights
    for (size_t i = 0; i < filter_weights_dL_dW.size(); i++) {
        for (size_t j = 0; j < std::get<0>(filter_weights_dL_dW[i].shape()); j++) {
            for (size_t k = 0; k < std::get<1>(filter_weights_dL_dW[i].shape()); k++) {
                for (size_t l = 0; l < std::get<2>(filter_weights_dL_dW[i].shape()); l++) {
                    const float gradient = filter_weights_dL_dW[i].at(j, k, l);
                    const float update = get_update(gradient, filter_weights_m_t_prev[i].at(j, k, l), filter_weights_v_t_prev[i].at(j, k, l), beta_1, beta_2, learning_rate, epsilon, adam_t);
                    // Apply updates
                    filter_weights[i].set(j, k, l, filter_weights[i].at(j, k, l) - update);
                }
            }
        }
    }
    
    // Filter biases
    for (size_t i = 0; i < filter_biases_dL_dB.size(); i++) {
        const float gradient = filter_biases_dL_dB[i];
        const float update = get_update(gradient, filter_biases_m_t_prev[i], filter_biases_v_t_prev[i], beta_1, beta_2, learning_rate, epsilon, adam_t);
        // Apply updates
        filter_biases[i] -= update;
    }

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
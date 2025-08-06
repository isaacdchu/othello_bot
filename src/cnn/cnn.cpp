#include "cnn.h"

void CNN::train(const std::string &data_path) {
    std::string line;
    std::ifstream data(data_path); // Open the training data file
    if (!data) {
        throw std::runtime_error("Failed to open data file: " + data_path);
    }
    while (std::getline(data, line)) {
        auto [input_tensor, value] = parse_line(line); // Parse the input tensor and value
        Output output = predict(input_tensor); // Get the CNN's prediction
        Board board = Board(input_tensor);
        MCTSNode root = MCTSNode(0, board, nullptr, true); // Black is the root player
        root.initialize_children(); // Initialize children nodes based on legal moves
        unsigned int iterations = 0;
        const unsigned int max_iterations = 100000;
        const unsigned int num_simulations = 10;
        float average_simulation_time = 0.0f;
        for (unsigned int i = 0; i < max_iterations; i++) {
            auto node = root.select();
            if (node == nullptr) break;
            for (unsigned int j = 0; j < num_simulations; j++) {
                const float result = node->simulate();
                node->backpropagate(result);
            }
        }
        std::array<float, 64> policy = root.get_policy(); // Get the policy distribution from MCTS
        Output target = {Tensor<8, 8, 1>(policy), Tensor<1, 1, 1>({static_cast<float>(value)})}; // Create the target output
        backpropagate(output, target); // Update the CNN based on the prediction and label
    }
}

Output CNN::predict(const std::string &data_path) const { 
    // Placeholder for predicting from a file
    return {
        Tensor<8, 8, 1>(),
        Tensor<1, 1, 1>()
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
    Tensor<1, 1, 1> value_output = dense_value(flattened_result);
    return {policy_output, value_output}; // Return the output structure
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
            float sum_1 = flatten_biases[0];
            float sum_2 = flatten_biases[1];
            for (size_t z = 0; z < 32; z++) {
                sum_1 += input.at(x, y, z) * flatten_weights[0].at(0, 0, z);
                sum_2 += input.at(x, y, z) * flatten_weights[1].at(0, 0, z);
            }
            output_tensor.set(x, y, 0, activation_function(sum_1));
            output_tensor.set(x, y, 1, activation_function(sum_2));
        }
    }
    return output_tensor;
}

Tensor<8, 8, 1> CNN::dense_policy(const Tensor<8, 8, 2> &input) const {
    // Dense layer to policy head
    std::array<float, 128> input_array = input.get_data();
    std::array<float, 64> output_array = {};
    for (size_t i = 0; i < 64; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < 128; j++) {
            sum += input_array[j] * policy_weights[i * 128 + j] + policy_biases[i];
        }
        output_array[i] = sum;
    }
    return Tensor<8, 8, 1>(softmax(output_array));
}

Tensor<1, 1, 1> CNN::dense_value(const Tensor<8, 8, 2> &input) const {
    // Dense layer to value head
    std::array<float, 128> input_array = input.get_data();
    float sum = value_bias; // Start with the bias
    for (size_t i = 0; i < 128; i++) {
        sum += input_array[i] * value_weights[i];
    }
    std::array<float, 1> output_array = {std::tanh(sum)};
    return Tensor<1, 1, 1>(output_array);
}

void CNN::backpropagate(const Output output, const Output label) {
    // TODO
    // Using Adam optimizer for backpropagation
}

std::pair<Tensor<8, 8, 3>, const int> CNN::parse_line(const std::string &line) {
    // Parses a line from the training data file
    std::array<float, 8 * 8 * 3> init_data = {};
    int evaluation = std::stoi(line.substr(384)); // Extract evaluation label from the line
    for (size_t i = 0; i < 383; i++) {
        if (line[i] == ',') continue; // Skip commas
        init_data[i] = line[i] - '0'; // Convert character to float
    }
    Tensor<8, 8, 3> tensor = Tensor<8, 8, 3>(init_data);
    return {tensor, evaluation};
}
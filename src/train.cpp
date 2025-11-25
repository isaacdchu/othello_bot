#include "cnn/cnn.hpp"
#include "cnn/tensor3d.hpp"
#include "cnn/optimizer.hpp"
#include "cnn/adam.hpp"

#include <iostream>
#include <string>
#include <fstream>

Tensor3D<8, 8, 3> parse_line(const std::string& line) {
    Tensor3D<8, 8, 3> tensor = Tensor3D<8, 8, 3>(0.0f);
    for (size_t i = 0; i < tensor.size(); i++) {
        const char c = line[2 * i];
        tensor.at(i) = static_cast<float>(c - '0');
    }
    return tensor;
}

float get_label(const std::string& line) {
    return static_cast<float>(line.back() - '0');
}

int main() {
    CNN<32, Adam, float, float, float> model(0.001f, 0.9f, 0.999f);
    std::ifstream infile("data/00.txt");
    std::string line;
    if (!infile.is_open()) {
        std::cerr << "Failed to open data/00.txt" << std::endl;
        return 1;
    }
    std::string first_line;
    if (std::getline(infile, line)) {
        first_line = line;
    }
    const int MAX_LINES = 1000;
    int line_count = 0;
    const int EPOCHS = 10;
    const int BATCH_SIZE = 10;
    std::vector<std::string> lines;

    // Read all lines into memory
    while (std::getline(infile, line) && line_count < MAX_LINES) {
        lines.push_back(line);
        line_count++;
    }

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << std::endl;
        for (size_t i = 0; i < lines.size(); i += BATCH_SIZE) {
            for (size_t j = i; j < i + BATCH_SIZE && j < lines.size(); ++j) {
                Tensor3D<8, 8, 3> input = parse_line(lines[j]);
                float label = get_label(lines[j]);
                Tensor3D<1, 1, 1> output = model.forward(input);
                model.backward(output - Tensor3D<1, 1, 1>(label));
            }
            model.update();
        }
    }
    Tensor3D<8, 8, 3> input = parse_line(first_line);
    float label = get_label(first_line);
    Tensor3D<1, 1, 1> output = model.forward(input);
    std::cout << "Predicted: " << output.at(0, 0, 0) << ", Actual: " << label << std::endl;
    model.save("models/cnn.model");
    return 0;
}

// TODO
// evaluation metrics
// integrate with MCTS and bot
// optimize performance (SIMD, CUDA)
// compare with PyTorch implementation
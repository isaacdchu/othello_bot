#include "cnn/cnn.hpp"
#include "cnn/tensor3d.hpp"
#include "cnn/optimizer.hpp"
#include "cnn/adam.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
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
    CNN<32, Adam, float, float, float> model("model/cnn.model");
    std::ifstream infile("data/01.txt");
    std::string line;
    if (!infile.is_open()) {
        std::cerr << "Failed to open data/01.txt" << std::endl;
        return 1;
    }

    const int MAX_LINES = 1000;
    std::vector<float> labels;
    std::vector<float> predictions;
    int line_count = 0;

    while (std::getline(infile, line) && line_count < MAX_LINES) {
        Tensor3D<8, 8, 3> input = parse_line(line);
        float label = get_label(line);
        Tensor3D<1, 1, 1> output = model.forward(input);
        labels.push_back(label);
        predictions.push_back(output.at(0, 0, 0));
        line_count++;
    }

    // Calculate Mean Squared Error
    float mse = 0.0f;
    const float n_samples = static_cast<float>(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        const float diff = labels[i] - predictions[i];
        mse += diff * diff / n_samples;
    }
    std::cout << "Label range: [" << *std::min_element(labels.begin(), labels.end()) << ", " << *std::max_element(labels.begin(), labels.end()) << "]" << std::endl;
    std::cout << "Prediction range: [" << *std::min_element(predictions.begin(), predictions.end()) << ", " << *std::max_element(predictions.begin(), predictions.end()) << "]" << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;
    return 0;
}

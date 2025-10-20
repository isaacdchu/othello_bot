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
    const char c = line.back();
    return static_cast<float>(c - '0');
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
    while (std::getline(infile, line) && line_count < MAX_LINES) {
        Tensor3D<8, 8, 3> input = parse_line(line);
        float label = get_label(line);
        Tensor3D<1, 1, 1> output = model.forward(input);
        model.backward(output - Tensor3D<1, 1, 1>(label));
        model.update();
        line_count++;
    }
    Tensor3D<8, 8, 3> input = parse_line(first_line);
    float label = get_label(first_line);
    Tensor3D<1, 1, 1> output = model.forward(input);
    std::cout << "Predicted: " << output.at(0, 0, 0) << ", Actual: " << label << std::endl;
    return 0;
}
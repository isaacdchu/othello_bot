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
    std::unique_ptr<Optimizer> optimizer = Adam::factory(0.001f, 0.9f, 0.999f);
    CNN<32> model = CNN<32>(*optimizer);
    std::ifstream infile("data/00.txt");
    std::string line;
    if (!infile.is_open()) {
        std::cerr << "Failed to open data/00.txt" << std::endl;
        return 1;
    }
    if (std::getline(infile, line)) {
        Tensor3D<8, 8, 3> input = parse_line(line);
        // std::cout << "Parsed input tensor: " << input.to_string() << std::endl;
        Tensor3D<1, 1, 1> output = model.forward(input);
        // std::cout << "Parsed input forward pass completed." << std::endl;
        std::cout << "Parsed output tensor: " << output.to_string() << std::endl;
        model.backward(Tensor3D<1, 1, 1>(8.0f));
        output = model.forward(input);
        std::cout << "Parsed output tensor: " << output.to_string() << std::endl;
    }
    return 0;
}
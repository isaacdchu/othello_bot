#include <iostream>

#include "cnn/cnn.hpp"

int main() {
    CNN<16> model = CNN<16>();
    Tensor3D<8, 8, 3> input = Tensor3D<8, 8, 3>(1.0f);
    Tensor3D<1, 1, 1> output = model.forward(input);
    std::cout << "Forward pass completed." << std::endl;
    std::cout << "Output tensor: " << output.to_string() << std::endl;
    return 0;
}
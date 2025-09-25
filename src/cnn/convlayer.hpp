#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "layer.hpp"

template<size_t X_in, size_t Y_in, size_t C_in, size_t K, size_t S, size_t P, size_t C_out>
class ConvLayer : public Layer<X_in, Y_in, C_in, X_out = (X_in - K + 2 * P) / S + 1, Y_out = (Y_in - K + 2 * P) / S + 1, C_out> {
public:
    ConvLayer() {
        // Initialize weights and biases
    }

    Tensor3D<X_out, Y_out, C_out> forward(const Tensor3D<X_in, Y_in, C_in>& input) override {
        // Implement forward pass
    }

    Tensor3D<X_in, Y_in, C_in> backward(const Tensor3D<X_out, Y_out, C_out>& grad_output) override {
        // Implement backward pass
    }
};

#endif // CONV_LAYER_HPP
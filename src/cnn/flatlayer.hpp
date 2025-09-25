#ifndef FLATLAYER_HPP
#define FLATLAYER_HPP

#include "layer.hpp"

template<size_t X_in, size_t Y_in, size_t C_in, size_t N_out>
class FlatLayer : public Layer<X_in, Y_in, C_in, N_out, 1, 1> {
public:
    FlatLayer() {
        // Initialize any necessary parameters
    }
    Tensor3D<N_out, 1, 1> forward(const Tensor3D<X_in, Y_in, C_in>& input) override {
        // Implement forward pass: flatten the input tensor
    }
    Tensor3D<X_in, Y_in, C_in> backward(const Tensor3D<N_out, 1, 1>& grad_output) override {
        // Implement backward pass: reshape the gradient to match input dimensions
    }
};

#endif // FLATLAYER_HPP
#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"

template<size_t N_in, size_t N_out>
class DenseLayer : public Layer<N_in, 1, 1, N_out, 1, 1> {
public:
    DenseLayer() {
        // Initialize weights and biases
    }

    Tensor3D<N_out, 1, 1> forward(const Tensor3D<N_in, 1, 1>& input) override {
        // Implement forward pass
    }

    Tensor3D<N_in, 1, 1> backward(const Tensor3D<N_out, 1, 1>& grad_output) override {
        // Implement backward pass
    }
};

#endif // DENSE_LAYER_HPP

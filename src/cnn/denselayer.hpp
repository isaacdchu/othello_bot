#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include <memory>

template<size_t N_in, size_t N_out>
class DenseLayer : public Layer<N_in, 1, 1, N_out, 1, 1> {
private:
    // Weights and biases for the dense layer
    Tensor3D<N_in, N_out, 1> weights; // Weight matrix
    Tensor3D<N_out, 1, 1> biases; // Biases for each output neuron
public:
    DenseLayer() {
        // Initialize weights and biases
    }

    std::unique_ptr<TensorInterface> forward(const TensorInterface& input) override {
        Tensor3D<N_out, 1, 1> output;
        for (size_t j = 0; j < N_out; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < N_in; i++) {
                sum += input.at(i, 0, 0) * weights.at(i, j, 0);
            }
            output.at(j, 0, 0) = sum + biases.at(j, 0, 0);
        }
        // Activation (ReLU)
        for (size_t j = 0; j < N_out; j++) {
            output.at(j, 0, 0) = std::max(0.0f, output.at(j, 0, 0));
        }
        return std::make_unique<Tensor3D<N_out, 1, 1>>(output);
    }

    std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) override {
        (void)grad_output; // suppress unused-parameter warning until implemented
        Tensor3D<N_in, 1, 1> grad_input = Tensor3D<N_in, 1, 1>(0.0f);
        return std::make_unique<Tensor3D<N_in, 1, 1>>(grad_input);
    }
};

#endif // DENSE_LAYER_HPP

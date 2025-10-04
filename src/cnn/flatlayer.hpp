#ifndef FLATLAYER_HPP
#define FLATLAYER_HPP

#include "layer.hpp"
#include "tensor3d.hpp"
#include "tensorinterface.hpp"

#include <memory>

template<size_t X_in, size_t Y_in, size_t C_in, size_t N_out>
class FlatLayer : public Layer<X_in, Y_in, C_in, N_out, 1, 1> {
private:
    // No weights or biases needed for flattening layer
public:
    FlatLayer() {
        // Initialize any necessary parameters
    }

    std::unique_ptr<TensorInterface> forward(const TensorInterface& input) override {
        // Reshape input tensor to output tensor
        Tensor3D<N_out, 1, 1> output;
        for (size_t i = 0; i < input.size(); i++) {
            output.at(i) = input.at(i);
        }
        return std::make_unique<Tensor3D<N_out, 1, 1>>(output);
    }

    std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) override {
        // Gradient w.r.t. input is reshaped gradient from output
        Tensor3D<X_in, Y_in, C_in> grad_input = Tensor3D<X_in, Y_in, C_in>(0.0f);
        for (size_t i = 0; i < grad_output.size(); i++) {
            grad_input.at(i) = grad_output.at(i);
        }
        return std::make_unique<Tensor3D<X_in, Y_in, C_in>>(grad_input);
    }
};

#endif // FLATLAYER_HPP
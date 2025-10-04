#ifndef FLATLAYER_HPP
#define FLATLAYER_HPP

#include "layer.hpp"
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
        // cast input to Tensor3D concrete if needed (you can read via TensorInterface methods directly)
        Tensor3D<N_out, 1, 1> output;
        for (size_t c = 0; c < C_in; c++) {
            for (size_t y = 0; y < Y_in; y++) {
                for (size_t x = 0; x < X_in; x++) {
                    size_t index = c * (X_in * Y_in) + y * X_in + x;
                    output.at(index, 0, 0) = input.at(x, y, c);
                }
            }
        }
        return std::make_unique<Tensor3D<N_out, 1, 1>>(output);
    }

    std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) override {
        (void)grad_output; // suppress unused-parameter warning until implemented
        Tensor3D<X_in, Y_in, C_in> grad_input = Tensor3D<X_in, Y_in, C_in>(0.0f);
        return std::make_unique<Tensor3D<X_in, Y_in, C_in>>(grad_input);
    }
};

#endif // FLATLAYER_HPP
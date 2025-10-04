#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "layer.hpp"
#include "tensor3d.hpp"
#include <memory>
#include <algorithm>

// X/Y are spatial dimensions of the board, C is the number of channels, K is kernel size, S is stride, P is padding
template<size_t X_in, size_t Y_in, size_t C_in, size_t K, size_t S, size_t P, size_t C_out>
class ConvLayer : public Layer<X_in, Y_in, C_in, (X_in - K + 2 * P) / S + 1, (Y_in - K + 2 * P) / S + 1, C_out> {
private:
    static constexpr size_t X_out = (X_in - K + 2 * P) / S + 1;
    static constexpr size_t Y_out = (Y_in - K + 2 * P) / S + 1;
    // Weights and biases for the convolutional layer
    Tensor3D<K, K, C_in * C_out> weights; // Convolutional filters
    Tensor3D<1, 1, C_out> biases; // Biases for each output channel
public:
    ConvLayer() {
        // Initialize weights and biases
        weights = Tensor3D<K, K, C_in * C_out>(0.01f); // Small random values
        biases = Tensor3D<1, 1, C_out>(0.0f);
    }

    // change signature to return a unique_ptr<TensorInterface>
    std::unique_ptr<TensorInterface> forward(const TensorInterface& input) override {
        Tensor3D<X_out, Y_out, C_out> output;
        for (size_t c_out = 0; c_out < C_out; c_out++) {
            for (size_t y = 0; y < Y_out; y++) {
                for (size_t x = 0; x < X_out; x++) {
                    float conv_sum = 0.0f;
                    for (size_t c_in = 0; c_in < C_in; c_in++) {
                        for (size_t ky = 0; ky < K; ky++) {
                            for (size_t kx = 0; kx < K; kx++) {
                                int in_x = x * S + kx - P;
                                int in_y = y * S + ky - P;
                                if (in_x >= 0 && in_y >= 0) {
                                    if ((unsigned)in_x < X_in && (unsigned)in_y < Y_in) {
                                        conv_sum += input.at(in_x, in_y, c_in) * weights.at(kx, ky, c_in + c_out * C_in);
                                    }
                                }
                            }
                        }
                    }
                    // Add bias
                    output.at(x, y, c_out) = conv_sum + biases.at(0, 0, c_out);
                }
            }
        }
        // Activation (ReLU)
        for (size_t c_out = 0; c_out < C_out; c_out++) {
            for (size_t y = 0; y < Y_out; y++) {
                for (size_t x = 0; x < X_out; x++) {
                    output.at(x, y, c_out) = std::max(0.0f, output.at(x, y, c_out));
                }
            }
        }
        return std::make_unique<Tensor3D<X_out, Y_out, C_out>>(output);
    }

    std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) override {
        (void)grad_output; // suppress unused-parameter warning until implemented
        Tensor3D<X_in, Y_in, C_in> grad_input = Tensor3D<X_in, Y_in, C_in>(0.0f);
        return std::make_unique<Tensor3D<X_in, Y_in, C_in>>(grad_input);
    }
};

#endif // CONV_LAYER_HPP
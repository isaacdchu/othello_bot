#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "layer.hpp"
#include "tensor3d.hpp"
#include "activation.hpp"
#include "convolution.hpp"
#include "optimizer.hpp"

#include <memory>
#include <algorithm>
#include <vector> 

// X/Y are spatial dimensions of the board, C is the number of channels, K is kernel size, S is stride, P is padding
template<size_t X_in, size_t Y_in, size_t C_in, size_t K, size_t S, size_t P, size_t C_out, template<auto> class Opt, typename... Args>
class ConvLayer : public Layer<X_in, Y_in, C_in, (X_in - K + 2 * P) / S + 1, (Y_in - K + 2 * P) / S + 1, C_out, Opt, Args...> {
private:
    static constexpr size_t X_out = (X_in - K + 2 * P) / S + 1;
    static constexpr size_t Y_out = (Y_in - K + 2 * P) / S + 1;
    // Weights and biases for the convolutional layer
    Tensor3D<K, K, C_in * C_out> weights; // Convolutional filters
    Tensor3D<1, 1, C_out> biases; // Biases for each output channel

    // Backpropagation storage
    // input tensor from last forward pass
    Tensor3D<X_in, Y_in, C_in> last_input;
    // f'(z) for each output neuron
    Tensor3D<X_out, Y_out, C_out> pre_activation_output;

    // Update storage
    Tensor3D<K, K, C_in * C_out> dL_dW;
    Tensor3D<1, 1, C_out> dL_db;

    // Optimizer
    Opt<decltype(dL_dW)::shape()> weights_optimizer;
    Opt<decltype(dL_db)::shape()> biases_optimizer;
public:
    explicit ConvLayer(Args... args)
        : weights(Tensor3D<K, K, C_in * C_out>(0.01f)),
          biases(Tensor3D<1, 1, C_out>(0.0f)),
          weights_optimizer(Opt<decltype(dL_dW)::shape()>(args...)),
          biases_optimizer(Opt<decltype(dL_db)::shape()>(args...)) {
        // initialize weights, biases, optimizers
        dL_dW = Tensor3D<K, K, C_in * C_out>(0.0f);
        dL_db = Tensor3D<1, 1, C_out>(0.0f);
    }

    explicit ConvLayer(const ConvLayer& other)
        : Layer<X_in, Y_in, C_in, X_out, Y_out, C_out, Opt, Args...>(other), weights(other.weights), biases(other.biases) {}

    explicit ConvLayer(const std::string& serialized_data) 
        : weights_optimizer(Opt<decltype(dL_dW)::shape()>()),
          biases_optimizer(Opt<decltype(dL_db)::shape()>()) {
        // Deserialize weights and biases from string
        size_t weights_pos = serialized_data.find("Weights:") + 8;
        size_t biases_pos = serialized_data.find("Biases:") + 7;
        size_t weights_end = serialized_data.find("\n", weights_pos);
        std::string weights_str = serialized_data.substr(weights_pos, weights_end - weights_pos);
        std::string biases_str = serialized_data.substr(biases_pos);
        weights = Tensor3D<K, K, C_in * C_out>(weights_str);
        biases = Tensor3D<1, 1, C_out>(biases_str);
        dL_dW = Tensor3D<K, K, C_in * C_out>(0.0f);
        dL_db = Tensor3D<1, 1, C_out>(0.0f);
    }
    
    std::unique_ptr<TensorInterface> forward(const TensorInterface& input) override {
        // cast to concrete tensor implementation (assumes caller passes Tensor3D)
        const auto& in = static_cast<const Tensor3D<X_in, Y_in, C_in>&>(input);
        last_input = in;
        Tensor3D<X_out, Y_out, C_out> output = convolute<X_in, Y_in, C_in, K, S, P, C_out>(in, weights, biases);
        pre_activation_output = output;
        grad_relu(pre_activation_output);
        relu(output);
        return std::make_unique<Tensor3D<X_out, Y_out, C_out>>(output);
    }

    std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) override {
        weights_optimizer.step();
        biases_optimizer.step();
        
        // Calculate dL/dW, dL/db, and dL/dInput

        // delta = dL/dz = dL/dOutput * f'(z)
        Tensor3D<X_out, Y_out, C_out> delta;
        for (size_t k = 0; k < C_out; k++) {
            for (size_t j = 0; j < Y_out; j++) {
                for (size_t i = 0; i < X_out; i++) {
                    delta.at(i, j, k) = grad_output.at(i, j, k) * pre_activation_output.at(i, j, k);
                }
            }
        }
        // dL/dW = input conv delta
        dL_dW += grad_convolute<X_in, Y_in, C_in, K, S, P, C_out>(last_input, delta);

        // dL/dbk = sum(i j) delta(i j k)
        for (size_t k = 0; k < C_out; k++) {
            float sum = 0.0f;
            for (size_t j = 0; j < Y_out; j++) {
                for (size_t i = 0; i < X_out; i++) {
                    sum += delta.at(i, j, k);
                }
            }
            dL_db.at(0, 0, k) += sum;
        }

        // dL/dInput = delta conv W_flipped
        Tensor3D<K, K, C_in * C_out> weights_flipped = Tensor3D<K, K, C_in * C_out>();
        // Flip weights for convolution
        for (size_t k = 0; k < C_out; k++) {
            for (size_t c = 0; c < C_in; c++) {
                for (size_t j = 0; j < K; j++) {
                    for (size_t i = 0; i < K; i++) {
                        weights_flipped.at(K - 1 - i, K - 1 - j, c + k * C_in) = weights.at(i, j, c + k * C_in);
                    }
                }
            }
        }
        Tensor3D<X_in, Y_in, C_in> dL_dInput = convolute<X_out, Y_out, C_out, K, 1, K - 1 - P, C_in>(delta, weights_flipped);

        return std::make_unique<Tensor3D<X_in, Y_in, C_in>>(dL_dInput);
    }

    void update() {
        weights_optimizer.update(weights, dL_dW);
        biases_optimizer.update(biases, dL_db);
        dL_dW.fill(0.0f);
        dL_db.fill(0.0f);
    }

    std::string to_string(bool details = false) const override {
        std::string str = "ConvLayer(" + std::to_string(X_in) + "x" + std::to_string(Y_in) + "x" + std::to_string(C_in) +
               " -> " + std::to_string(X_out) + "x" + std::to_string(Y_out) + "x" + std::to_string(C_out) + ")" +
               " [K=" + std::to_string(K) + ", S=" + std::to_string(S) + ", P=" + std::to_string(P) + "]";
        if (!details) return str;
        str += "\nWeights:\n" + weights.to_string() + "\nBiases:\n" + biases.to_string();
        return str;
    }

    LayerInterface<Opt, Args...>* clone() const override {
        return new ConvLayer<X_in, Y_in, C_in, K, S, P, C_out, Opt, Args...>(*this);
    }

    std::string serialize() const override {
        std::string data = "ConvLayer\n";
        data += "Weights:" + weights.serialize() + "\n";
        data += "Biases:" + biases.serialize() + "\n";
        return data;
    }
};

#endif // CONV_LAYER_HPP
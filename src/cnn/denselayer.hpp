#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "tensor3d.hpp"
#include "tensorinterface.hpp"
#include "activation.hpp"

#include <memory>

template<size_t N_in, size_t N_out, template<auto> class Opt, typename... Args>
class DenseLayer : public Layer<N_in, 1, 1, N_out, 1, 1, Opt, Args...> {
private:
    // Weights and biases for the dense layer
    Tensor3D<N_in, N_out, 1> weights; // Weight matrix
    Tensor3D<N_out, 1, 1> biases; // Biases for each output neuron

    // Backpropagation storage
    // input tensor from last forward pass
    Tensor3D<N_in, 1, 1> last_input;
    // f'(z) for each output neuron
    Tensor3D<N_out, 1, 1> pre_activation_output;

    // Update storage
    decltype(weights) dL_dW;
    decltype(biases) dL_db;

    Opt<decltype(weights)::shape()> weights_optimizer;
    Opt<decltype(biases)::shape()> biases_optimizer;

public:
    explicit DenseLayer(Args... args) 
        // Initialize weights and biases
        : weights(Tensor3D<N_in, N_out, 1>(0.01f)), // Small random values
          biases(Tensor3D<N_out, 1, 1>(0.0f)),
          pre_activation_output(Tensor3D<N_out, 1, 1>(0.0f)),
          weights_optimizer(Opt<decltype(weights)::shape()>(args...)),
          biases_optimizer(Opt<decltype(biases)::shape()>(args...)) {
        // initialize weights, biases, optimizers
        dL_dW = Tensor3D<N_in, N_out, 1>(0.0f);
        dL_db = Tensor3D<N_out, 1, 1>(0.0f);
    }

    explicit DenseLayer(const DenseLayer& other) 
        : Layer<N_in, 1, 1, N_out, 1, 1, Opt, Args...>(other), weights(other.weights), biases(other.biases), last_input(other.last_input), pre_activation_output(other.pre_activation_output) {}

    DenseLayer(const std::string& serialized_data) 
        : weights_optimizer(Opt<decltype(weights)::shape()>()),
          biases_optimizer(Opt<decltype(biases)::shape()>()) {
        // Deserialize weights and biases from string
        size_t weights_pos = serialized_data.find("Weights:") + 8;
        size_t biases_pos = serialized_data.find("Biases:") + 7;
        size_t weights_end = serialized_data.find("\n", weights_pos);
        std::string weights_str = serialized_data.substr(weights_pos, weights_end - weights_pos);
        std::string biases_str = serialized_data.substr(biases_pos);
        weights = Tensor3D<N_in, N_out, 1>(weights_str);
        biases = Tensor3D<N_out, 1, 1>(biases_str);
    }

    std::unique_ptr<TensorInterface> forward(const TensorInterface& input) override {
        last_input = Tensor3D<N_in, 1, 1>(input);
        Tensor3D<N_out, 1, 1> output;
        for (size_t j = 0; j < N_out; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < N_in; i++) {
                sum += input.at(i, 0, 0) * weights.at(i, j, 0);
            }
            output.at(j, 0, 0) = sum + biases.at(j, 0, 0);
        }
        // Calculate f'(z) for backpropagation
        pre_activation_output = output;
        grad_relu(pre_activation_output);
        relu(output);
        return std::make_unique<Tensor3D<N_out, 1, 1>>(output);
    }

    std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) override {
        weights_optimizer.step();
        biases_optimizer.step();
        
        // Calculate dL/dW, dL/db, and dL/dInput
        // delta = grad_output * f'(z)
        Tensor3D<N_out, 1, 1> delta;
        for (size_t i = 0; i < N_out; i++) {
            delta.at(i) = grad_output.at(i) * pre_activation_output.at(i);
        }

        // dL/dW = delta * input^T
        for (size_t i = 0; i < N_in; i++) {
            for (size_t j = 0; j < N_out; j++) {
                dL_dW.at(i, j, 0) += delta.at(j) * last_input.at(i);
            }
        }

        // dL/db = delta
        dL_db += delta;

        // dL/dInput = W^T * delta
        Tensor3D<N_in, 1, 1> dL_dInput;
        for (size_t i = 0; i < N_in; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < N_out; j++) {
                sum += weights.at(i, j, 0) * delta.at(j);
            }
            dL_dInput.at(i) = sum;
        }

        return std::make_unique<Tensor3D<N_in, 1, 1>>(dL_dInput);
    }

    void update() override {
        weights_optimizer.update(weights, dL_dW);
        biases_optimizer.update(biases, dL_db);
        dL_dW.fill(0.0f);
        dL_db.fill(0.0f);
    }

    std::string to_string(bool details = false) const override {
        std::string str = "DenseLayer<" + std::to_string(N_in) + ", " + std::to_string(N_out) + ">";
        if (!details) return str;
        str += "\nWeights:\n" + weights.to_string() + "\nBiases:\n" + biases.to_string();
        return str;
    }

    LayerInterface<Opt, Args...>* clone() const override {
        return new DenseLayer<N_in, N_out, Opt, Args...>(*this);
    }

    std::string serialize() const override {
        std::string data = "DenseLayer\n";
        data += "Weights:" + weights.serialize() + "\n";
        data += "Biases:" + biases.serialize() + "\n";
        return data;
    }
};

#endif // DENSE_LAYER_HPP

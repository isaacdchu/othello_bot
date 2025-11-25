#ifndef FLATLAYER_HPP
#define FLATLAYER_HPP

#include "layer.hpp"
#include "tensor3d.hpp"
#include "tensorinterface.hpp"

#include <memory>

template<size_t X_in, size_t Y_in, size_t C_in, size_t N_out, template<auto> class Opt, typename... Args>
class FlatLayer : public Layer<X_in, Y_in, C_in, N_out, 1, 1, Opt, Args...> {
private:
    // No weights or biases needed for flattening layer
public:
    explicit FlatLayer(Args...) {
        // no parameters to initialize
    }
    
    explicit FlatLayer(const FlatLayer& other) : Layer<X_in, Y_in, C_in, N_out, 1, 1, Opt, Args...>(other) {}

    FlatLayer(const std::string& serialized_data) {
        // no parameters to deserialize
        (void)serialized_data; // suppress unused-parameter warning
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

    void update() override {
        // do nothing, no parameters to update
    }

    std::string to_string(bool details = false) const override {
        (void)details; // suppress unused-parameter warning
        std::string str = "FlatLayer(" + std::to_string(X_in) + "x" + std::to_string(Y_in) + "x" + std::to_string(C_in) +
               " -> " + std::to_string(N_out) + "x1x1)";
        return str;
    }

    LayerInterface<Opt, Args...>* clone() const override {
        return new FlatLayer<X_in, Y_in, C_in, N_out, Opt, Args...>(*this);
    }

    std::string serialize() const override {
        return "FlatLayer\n"; // No parameters to serialize
    }
};

#endif // FLATLAYER_HPP
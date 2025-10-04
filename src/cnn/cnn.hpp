#ifndef CNN_HPP
#define CNN_HPP

#include "layer.hpp"
#include "convlayer.hpp"
#include "flatlayer.hpp"
#include "denselayer.hpp"
#include "adam.hpp"

#include <vector>
#include <array>
#include <memory>
#include <stdexcept>

template<size_t num_filters>
class CNN {
private:
    static constexpr size_t filter_size = 3;
    static constexpr size_t stride = 1;
    static constexpr size_t padding = 1;
    std::vector<std::unique_ptr<LayerInterface>> layers;

public:
    CNN() {
        std::unique_ptr<ConvLayer<8, 8, 3, filter_size, stride, stride, num_filters>> conv_layer = std::make_unique<ConvLayer<8, 8, 3, filter_size, stride, stride, num_filters>>();
        std::unique_ptr<FlatLayer<8, 8, num_filters, 8 * 8 * num_filters>> flat_layer = std::make_unique<FlatLayer<8, 8, num_filters, 8 * 8 * num_filters>>();
        std::unique_ptr<DenseLayer<8 * 8 * num_filters, 1>> dense_layer = std::make_unique<DenseLayer<8 * 8 * num_filters, 1>>();
        layers.push_back(std::move(conv_layer));
        layers.push_back(std::move(flat_layer));
        layers.push_back(std::move(dense_layer));
    }

    // Accept concrete input tensor (simpler and safe)
    Tensor3D<1, 1, 1> forward(const Tensor3D<8, 8, 3>& input) {
        std::unique_ptr<TensorInterface> current_input = std::make_unique<Tensor3D<8, 8, 3>>(input);
        for (const auto& layer : layers) {
            current_input = layer->forward(*current_input);
        }
        auto concrete = dynamic_cast<Tensor3D<1,1,1>*>(current_input.get());
        if (!concrete) throw std::runtime_error("CNN::forward: final layer did not return Tensor3D<1,1,1>");
        return *concrete;
    }
    
    // Backward: accept concrete gradient for output and propagate backwards, return gradient w.r.t. input
    Tensor3D<8, 8, 3> backward(const Tensor3D<1, 1, 1>& grad_output) {
        std::unique_ptr<TensorInterface> current_grad = std::make_unique<Tensor3D<1, 1, 1>>(grad_output);
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(*current_grad);
        }
        auto concrete = dynamic_cast<Tensor3D<8,8,3>*>(current_grad.get());
        if (!concrete) throw std::runtime_error("CNN::backward: final grad is not Tensor3D<8,8,3>");
        return *concrete;
    }
};

#endif // CNN_HPP
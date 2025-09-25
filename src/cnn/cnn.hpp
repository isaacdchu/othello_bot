#ifndef CNN_HPP
#define CNN_HPP

#include "layer.hpp"
#include "convlayer.hpp"
#include "flatlayer.hpp"
#include "denselayer.hpp"
#include "adam.hpp"

#include <vector>
#include <array>

template<size_t num_filters>
class CNN {
private:
    constexpr size_t filter_size = 3;
    constexpr size_t stride = 1;
    constexpr size_t padding = 1;
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

    TensorInterface forward(const TensorInterface& input) {
        std::unique_ptr<TensorInterface> current_input = std::make_unique<TensorInterface>(input);
        for (const auto& layer : layers) {
            current_input = std::make_unique<TensorInterface>(layer->forward(*current_input));
        }
        const TensorInterface output = *current_input;
        return output;
    }

    TensorInterface backward(const TensorInterface& grad_output) {
        std::unique_ptr<TensorInterface> current_grad = std::make_unique<TensorInterface>(grad_output);
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = std::make_unique<TensorInterface>((*it)->backward(*current_grad));
        }
        const TensorInterface output = *current_grad;
        return output;
    }
};

#endif // CNN_HPP
#ifndef CNN_HPP
#define CNN_HPP

#include "layer.hpp"
#include "convlayer.hpp"
#include "flatlayer.hpp"
#include "denselayer.hpp"
#include "optimizer.hpp"
#include "adam.hpp"

#include <vector>
#include <array>
#include <memory>
#include <stdexcept>

template<size_t num_filters, template<auto> class Opt, typename... Args>
class CNN {
private:
    static constexpr size_t filter_size = 3;
    static constexpr size_t stride = 1;
    static constexpr size_t padding = 1;
    std::vector<std::unique_ptr<LayerInterface<Opt, Args...>>> layers;

public:
    CNN(Args... args) {
        auto conv_layer = std::make_unique<ConvLayer<8, 8, 3, filter_size, stride, stride, num_filters, Opt, Args...>>(args...);
        auto flat_layer = std::make_unique<FlatLayer<8, 8, num_filters, 8 * 8 * num_filters, Opt, Args...>>(args...);
        auto dense_layer_1 = std::make_unique<DenseLayer<8 * 8 * num_filters, 512, Opt, Args...>>(args...);
        auto dense_layer_2 = std::make_unique<DenseLayer<512, 1, Opt, Args...>>(args...);
        layers.emplace_back(std::move(conv_layer));
        layers.emplace_back(std::move(flat_layer));
        layers.emplace_back(std::move(dense_layer_1));
        layers.emplace_back(std::move(dense_layer_2));
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

    void update() {
        for (const auto& layer : layers) {
            layer->update();
        }
    }

    std::vector<std::unique_ptr<LayerInterface<Opt, Args...>>> get_layers() const {
        std::vector<std::unique_ptr<LayerInterface<Opt, Args...>>> layer_refs;
        for (const auto& layer : layers) {
            layer_refs.push_back(std::unique_ptr<LayerInterface<Opt, Args...>>(layer->clone()));
        }
        return layer_refs;
    }
};

#endif // CNN_HPP
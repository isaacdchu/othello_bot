#ifndef CNN_HPP
#define CNN_HPP

#include "model.hpp"
#include "layer.hpp"
#include "dense_layer.hpp"

#include <cstddef>
#include <vector>
#include <functional>
#include <memory>

class CNN : public Model {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
public:
    CNN() {
        layers_.emplace_back(std::make_unique<DenseLayer>(std::vector<std::size_t>{192}, std::vector<std::size_t>{96}));
        layers_.emplace_back(std::make_unique<DenseLayer>(std::vector<std::size_t>{96}, std::vector<std::size_t>{1}));
    }

    void train() override {
        for (auto& layer : layers_) {
            layer->train();
        }
    }

    void eval() override {
        for (auto& layer : layers_) {
            layer->eval();
        }
    }

    Tensor forward(const Tensor& input) override {
        Tensor result = Tensor(input);
        for (auto& layer : layers_) {
            result = layer->forward(result);
        }
        return result;
    }

    std::vector<std::reference_wrapper<Tensor>> parameters() override {
        std::vector<std::reference_wrapper<Tensor>> params;
        for (auto& layer : layers_) {
            auto layer_params = layer->parameters();
            params.append_range(layer_params);
        }
        return params;
    }
};

#endif // CNN_HPP
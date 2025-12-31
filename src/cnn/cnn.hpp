#ifndef CNN_HPP
#define CNN_HPP

#include "model.hpp"
#include "layer.hpp"
#include "dense_layer.hpp"

#include <cstddef>
#include <vector>
#include <functional>

class CNN : public Model {
private:
    std::vector<Layer> layers_;
public:
    CNN() {
        layers_ = {
            DenseLayer({192}, {96}),
            DenseLayer({96}, {1}),
        };
    }

    void train() override {
        for (auto& layer : layers_) {
            layer.train();
        }
    }

    void eval() override {
        for (auto& layer : layers_) {
            layer.eval();
        }
    }

    Tensor forward(const Tensor& input) override {
        Tensor result = Tensor(input);
        for (auto& layer : layers_) {
            result = layer.forward(result);
        }
        return result;
    }

    std::vector<std::reference_wrapper<Tensor>> parameters() override {
        std::vector<std::reference_wrapper<Tensor>> params;
        for (auto& layer : layers_) {
            auto layer_params = layer.parameters();
            params.append_range(layer_params);
        }
        return params;
    }
};

#endif // CNN_HPP
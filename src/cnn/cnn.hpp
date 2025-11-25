#ifndef CNN_HPP
#define CNN_HPP

#include "model.hpp"
#include "layer.hpp"
#include "convlayer.hpp"
#include "flatlayer.hpp"
#include "denselayer.hpp"
#include "optimizer.hpp"
#include "adam.hpp"

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <stdexcept>

template <typename... Args>
std::tuple<Args...> parse_layer_args(const std::string& line) {
    std::istringstream ss(line);
    std::tuple<Args...> args;

    auto parse = [&ss](auto&... elements) {
        ((ss >> elements), ...);
    };

    std::apply(parse, args);
    return args;
}

template <size_t num_filters, template <auto> class Opt, typename... Args>
class CNN : public Model {
private:
    static constexpr size_t filter_size = 3;
    static constexpr size_t stride = 1;
    static constexpr size_t padding = 1;
    std::vector<std::unique_ptr<LayerInterface<Opt, Args...>>> layers;
    std::tuple<Args...> layer_args;

public:
    CNN(Args... args) : layer_args(args...) {
        auto conv_layer = std::make_unique<ConvLayer<8, 8, 3, filter_size, stride, stride, num_filters, Opt, Args...>>(args...);
        auto flat_layer = std::make_unique<FlatLayer<8, 8, num_filters, 8 * 8 * num_filters, Opt, Args...>>(args...);
        auto dense_layer_1 = std::make_unique<DenseLayer<8 * 8 * num_filters, 512, Opt, Args...>>(args...);
        auto dense_layer_2 = std::make_unique<DenseLayer<512, 1, Opt, Args...>>(args...);
        layers.emplace_back(std::move(conv_layer));
        layers.emplace_back(std::move(flat_layer));
        layers.emplace_back(std::move(dense_layer_1));
        layers.emplace_back(std::move(dense_layer_2));
    }

    CNN(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for loading CNN model: " + file_path);
        }
        std::string line;
        // Read layer args from first line
        if (std::getline(file, line)) {
            layer_args = parse_layer_args<Args...>(line);
        } else {
            throw std::runtime_error("Failed to read layer args from CNN model file: " + file_path);
        }
        // Read layers in order that they appear
        while (std::getline(file, line)) {
            if (line.find("ConvLayer") != std::string::npos) {
                std::string layer_data = "";
                while (std::getline(file, line) && !line.empty()) {
                    layer_data += line + "\n";
                }
                auto layer = std::make_unique<ConvLayer<8, 8, 3, filter_size, stride, stride, num_filters, Opt, Args...>>(layer_data);
                layers.emplace_back(std::move(layer));
            } else if (line.find("FlatLayer") != std::string::npos) {
                auto layer = std::make_unique<FlatLayer<8, 8, num_filters, 8 * 8 * num_filters, Opt, Args...>>(line);
                layers.emplace_back(std::move(layer));
            } else if (line.find("DenseLayer") != std::string::npos) {
                std::string layer_data = "";
                while (std::getline(file, line) && !line.empty()) {
                    layer_data += line + "\n";
                }
                if (layers.size() == 2) {
                    auto layer = std::make_unique<DenseLayer<8 * 8 * num_filters, 512, Opt, Args...>>(layer_data);
                    layers.emplace_back(std::move(layer));
                } else if (layers.size() == 3) {
                    auto layer = std::make_unique<DenseLayer<512, 1, Opt, Args...>>(layer_data);
                    layers.emplace_back(std::move(layer));
                } else {
                    throw std::runtime_error("Unexpected DenseLayer position in CNN model file: " + line);
                }
            }
        }
    }

    // Accept concrete input tensor (simpler and safe)
    Tensor3D<1, 1, 1> forward(const Tensor3D<8, 8, 3>& input) override {
        std::unique_ptr<TensorInterface> current_input = std::make_unique<Tensor3D<8, 8, 3>>(input);
        for (const auto& layer : layers) {
            current_input = layer->forward(*current_input);
        }
        auto concrete = dynamic_cast<Tensor3D<1,1,1>*>(current_input.get());
        if (!concrete) throw std::runtime_error("CNN::forward: final layer did not return Tensor3D<1,1,1>");
        return *concrete;
    }
    
    // Backward: accept concrete gradient for output and propagate backwards, return gradient w.r.t. input
    Tensor3D<8, 8, 3> backward(const Tensor3D<1, 1, 1>& grad_output) override {
        std::unique_ptr<TensorInterface> current_grad = std::make_unique<Tensor3D<1, 1, 1>>(grad_output);
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(*current_grad);
        }
        auto concrete = dynamic_cast<Tensor3D<8,8,3>*>(current_grad.get());
        if (!concrete) throw std::runtime_error("CNN::backward: final grad is not Tensor3D<8,8,3>");
        return *concrete;
    }

    void update() override {
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

    void save(const std::string& file_path) const override {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for saving CNN model: " + file_path);
        }
        std::apply([&file](const auto&... args) {
            ((file << args << " "), ...);
            file << std::endl;
        }, layer_args);
        for (const auto& layer : layers) {
            file << layer->serialize() << std::endl;
        }
    }
};

#endif // CNN_HPP
#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "dataset.hpp"
#include "../cnn/tensor.hpp"

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <memory>

class DataLoader {
private:
    const Dataset& dataset_;
    const std::size_t n_samples_;
    const std::size_t batch_size_;
    const std::size_t size_;
public:
    DataLoader(const Dataset& dataset, std::size_t n_samples = 0, std::size_t batch_size = 1) :
    dataset_(dataset),
    n_samples_(n_samples),
    batch_size_(batch_size),
    size_((n_samples == 0) ? dataset.size() / batch_size : n_samples / batch_size) {
        if (n_samples_ > dataset.size()) {
            throw std::invalid_argument("Number of samples exceeds dataset size.");
        }
        if (batch_size_ <= 0) {
            throw std::invalid_argument("Batch size must be greater than zero.");
        }
        if (size_ <= 0) {
            throw std::invalid_argument("Calculated size must be greater than zero.");
        }
    }

    std::pair<Tensor, Tensor> get_batch(std::size_t index) const {
        index *= batch_size_;
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        for (std::size_t i = 0; i < batch_size_; i++) {
            auto [input, target] = dataset_.get_item(index + i);
            inputs.push_back(input);
            targets.push_back(target);
        }
        Tensor batch_input = Tensor::stack(inputs);
        Tensor batch_target = Tensor::stack(targets);
        return {batch_input, batch_target};
    }

    std::size_t batch_size() const {
        return batch_size_;
    }

    std::size_t size() const {
        return size_;
    }
};

#endif // DATALOADER_HPP
#ifndef DATASET_HPP
#define DATASET_HPP

#include "../cnn/tensor.hpp"
#include <vector>
#include <cstddef>

class Dataset {
public:
    virtual std::pair<Tensor, Tensor> get_item(std::size_t index) const = 0;
    virtual std::size_t size() const = 0;
};

#endif // DATASET_HPP
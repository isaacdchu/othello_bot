#ifndef DATAHANDLER_HPP
#define DATAHANDLER_HPP

#include "dataset.hpp"

#include <cstddef>

class DataHandler {
private:
    const Dataset& dataset_;
public:
    DataHandler(const Dataset& dataset, std::size_t n_samples = 0, std::size_t batch_size = 1) : dataset_(dataset) {

    }

    std::pair<Tensor, Tensor> get_batch(std::size_t index) const {
        return dataset_.get_item(index);
    }

    std::size_t size() const {
        return dataset_.size();
    }
};

#endif // DATAHANDLER_HPP
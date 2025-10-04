#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "tensorinterface.hpp"
#include <algorithm>

void relu(TensorInterface& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.at(i) = std::max(0.0f, tensor.at(i));
    }
}

void grad_relu(TensorInterface& tensor) {
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.at(i) = (tensor.at(i) > 0.0f) ? 1.0f : 0.0f;
    }
}

#endif // ACTIVATION_HPP
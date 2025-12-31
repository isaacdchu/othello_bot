#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

#include <vector>
#include <functional>

class Layer {
public:
    virtual void train() = 0;
    virtual void eval() = 0;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<std::reference_wrapper<Tensor>> parameters() = 0;
};

#endif // LAYER_HPP
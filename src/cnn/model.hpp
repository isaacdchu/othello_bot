#ifndef MODEL_HPP
#define MODEL_HPP

#include "layer.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"

#include <vector>
#include <functional>

class Model {
public:
    virtual void train() = 0;
    virtual void eval() = 0;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<std::reference_wrapper<Tensor>> parameters() = 0;
};

#endif // MODEL_HPP
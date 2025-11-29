#ifndef MODEL_HPP
#define MODEL_HPP

#include "layer.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"

class Model {
public:
    virtual void train() = 0;
    virtual void eval() = 0;
    virtual Tensor forward(const Tensor& input) = 0;
};

#endif // MODEL_HPP
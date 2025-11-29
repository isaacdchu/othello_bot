#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

class Loss {
public:
    virtual double forward(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor backward() = 0;
};

#endif // LOSS_HPP
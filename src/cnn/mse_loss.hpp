#ifndef MSE_LOSS_HPP
#define MSE_LOSS_HPP

#include "loss.hpp"
#include "tensor.hpp"

class MSELoss : public Loss {
public:
    double forward(const Tensor& prediction, const Tensor& target) override {
        // Implement the forward pass for MSE loss
    }

    Tensor backward() override {
        // Implement the backward pass for MSE loss
    }
};

#endif // MSE_LOSS_HPP
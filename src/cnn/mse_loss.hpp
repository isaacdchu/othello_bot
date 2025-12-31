#ifndef MSE_LOSS_HPP
#define MSE_LOSS_HPP

#include "loss.hpp"
#include "tensor.hpp"

class MSELoss : public Loss {
public:
    Tensor forward(const Tensor& prediction, const Tensor& target) override {
        float res = 0.0f;
        for (std::size_t i = 0; i < prediction.size(); i++) {
            const float diff = prediction.at(i) - target.at(i);
            res += diff * diff;
        }
        res /= static_cast<float>(prediction.size());
        return Tensor({1}, {res});
    }
};

#endif // MSE_LOSS_HPP
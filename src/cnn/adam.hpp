#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizer.hpp"
#include "layerinterface.hpp"
#include "tensorinterface.hpp"
#include "tensor3d.hpp"

#include <array>
#include <memory>

#include <cmath>

template<std::array<size_t, 3> init_shape>
class Adam : public Optimizer<decltype(init_shape)> {
private:
    const float learning_rate;
    const float beta1;
    const float beta2;
    const float epsilon = 1e-8f;
    std::unique_ptr<TensorInterface> mt;
    std::unique_ptr<TensorInterface> vt;
    int t;
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f)
            : learning_rate(learning_rate), beta1(beta1), beta2(beta2), t(0) {
        // Initialize moment estimates
        Tensor3D<init_shape[0], init_shape[1], init_shape[2]> zero_tensor(0.0f);
        mt = std::make_unique<Tensor3D<init_shape[0], init_shape[1], init_shape[2]>>(zero_tensor);
        vt = std::make_unique<Tensor3D<init_shape[0], init_shape[1], init_shape[2]>>(zero_tensor);
    }

    void step() override {
        t++;
    }

    // mt = beta1 * mt-1 + (1 - beta1) * dL/dwt
    // vt = beta2 * vt-1 + (1 - beta2) * (dL/dwt)^2
    // mht = mt / (1 - beta1^t)
    // vht = vt / (1 - beta2^t)
    // wt+1 = wt - learning_rate * mht / (sqrt(vht) + epsilon)
    void update(TensorInterface& param, const TensorInterface& gradients) override {
        const float mhat_scale = (1.0f - std::pow(beta1, t));
        const float vhat_scale = (1.0f - std::pow(beta2, t));
        for (size_t i = 0; i < param.size(); i++) {
            mt->at(i) = beta1 * mt->at(i) + (1.0f - beta1) * gradients.at(i);
            vt->at(i) = beta2 * vt->at(i) + (1.0f - beta2) * gradients.at(i) * gradients.at(i);
            float mhat = mt->at(i) / mhat_scale;
            float vhat = vt->at(i) / vhat_scale;
            // Update parameters
            param.at(i) -= learning_rate * mhat / (std::sqrt(vhat) + epsilon);
        }
    }

    static std::unique_ptr<Optimizer> factory() {
        return std::make_unique<Adam>();
    }

    static std::unique_ptr<Optimizer> factory(const Adam& prototype) {
        return std::make_unique<Adam>(prototype.learning_rate, prototype.beta1, prototype.beta2);
    }

    static std::unique_ptr<Optimizer> factory(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f) {
        return std::make_unique<Adam>(learning_rate, beta1, beta2);
    }

    std::unique_ptr<Optimizer> clone() const override {
        return Adam::factory(*this);
    }
};

#endif // ADAM_HPP

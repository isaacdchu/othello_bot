#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizer.hpp"
#include "layerinterface.hpp"
#include "tensorinterface.hpp"

class Adam : public Optimizer {
private:
    const float learning_rate;
    const float beta1;
    const float beta2;
    const float epsilon = 1e-8f;
    float mt;
    float vt;
    int t;
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f)
            : learning_rate(learning_rate), beta1(beta1), beta2(beta2), mt(0), vt(0), t(0) {
        // Initialize moment estimates
    }

    // mt = beta1 * mt-1 + (1 - beta1) * dL/dwt
    // vt = beta2 * vt-1 + (1 - beta2) * (dL/dwt)^2
    // mht = mt / (1 - beta1^t)
    // vht = vt / (1 - beta2^t)
    // wt+1 = wt - learning_rate * mht / (sqrt(vht) + epsilon)
    void update(TensorInterface& param, const TensorInterface& gradients) override {
        t++;

    }

    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<Adam>(*this);
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
};

#endif // ADAM_HPP

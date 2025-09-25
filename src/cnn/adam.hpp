#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizer.hpp"

class Adam : public Optimizer {
private:
    const float learning_rate;
    const float beta1;
    const float beta2;
    const float epsilon = 1e-8f;
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f)
            : learning_rate(learning_rate), beta1(beta1), beta2(beta2) {
        // Initialize moment estimates
    }

    void update(LayerInterface& layer) override {
        // Implement Adam update logic here
    }
};

#endif // ADAM_HPP

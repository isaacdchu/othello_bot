#ifndef ADAM_H
#define ADAM_H

#include <cmath>

inline float get_update(const float gradient, float m_t_prev, float v_t_prev, const float beta_1, const float beta_2, const float learning_rate, const float epsilon, const unsigned int adam_t) {
    // Also updates m_t_prev and v_t_prev for loop compatibility
    const float m_t = beta_1 * m_t_prev + (1 - beta_1) * gradient;
    const float v_t = beta_2 * v_t_prev + (1 - beta_2) * gradient * gradient;
    m_t_prev = m_t;
    v_t_prev = v_t;
    const float m_hat = m_t / (1 - std::powf(beta_1, adam_t));
    const float v_hat = v_t / (1 - std::powf(beta_2, adam_t));
    if (m_hat != m_hat) {
        std::cout << m_t << "\n";
        std::cout << m_t_prev << "\n";
        std::cout << gradient << "\n";
        std::cout << std::powf(beta_1, adam_t) << std::endl;
        throw std::runtime_error("m_hat is bad");
    }
    if (v_hat != v_hat) {
        std::cout << v_t << "\n";
        std::cout << std::powf(beta_2, adam_t) << std::endl;
        throw std::runtime_error("v_hat is bad");
    }
    if (v_hat < 0) {
        std::cout << v_t << "\n";
        std::cout << std::powf(beta_2, adam_t) << std::endl;
        throw std::runtime_error("v_hat is negative");
    }
    const float update = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    if (update != update) {
        std::cout << adam_t << "\n";
        std::cout << m_t << "\n";
        std::cout << v_t << "\n";
        std::cout << m_t_prev << "\n";
        std::cout << v_t_prev << "\n";
        std::cout << gradient << "\n";
        std::cout << std::pow(beta_1, adam_t) << "\n";
        std::cout << std::pow(beta_2, adam_t) << "\n";
        std::cout << m_hat << "\n";
        std::cout << v_hat << "\n";
        std::cout << update << std::endl;
        throw std::runtime_error("Update is bad");
    }
    return update;
}

#endif // ADAM_H
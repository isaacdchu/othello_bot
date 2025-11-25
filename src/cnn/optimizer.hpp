#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "layerinterface.hpp"
#include "tensorinterface.hpp"

#include <memory>

template<typename... Args>
class Optimizer {
public:
    ~Optimizer() = default;
    virtual void step() = 0;
    virtual void update(TensorInterface& param, const TensorInterface& gradients) = 0;
    virtual std::unique_ptr<Optimizer> clone() const = 0;
    static std::unique_ptr<Optimizer> factory();
    static std::unique_ptr<Optimizer> factory(const Optimizer& prototype);
    static std::unique_ptr<Optimizer> factory(...);
};

#endif // OPTIMIZER_HPP
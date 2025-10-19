#ifndef LAYER_INTERFACE_HPP
#define LAYER_INTERFACE_HPP

#include "tensorinterface.hpp"
#include "optimizer.hpp"
#include <memory>

class LayerInterface {
protected:
    Optimizer& optimizer;
public:
    explicit LayerInterface(Optimizer& optimizer) : optimizer(optimizer) {}
    virtual ~LayerInterface() = default;
    virtual std::unique_ptr<TensorInterface> forward(const TensorInterface& input) = 0;
    virtual std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) = 0;
    virtual void update(const Optimizer& optimizer) = 0;
    virtual std::string to_string(bool details = false) const = 0;
    virtual LayerInterface* clone() const = 0;
};

#endif // LAYER_INTERFACE_HPP
#ifndef LAYER_INTERFACE_HPP
#define LAYER_INTERFACE_HPP

#include "tensorinterface.hpp"
#include <memory>

class LayerInterface {
public:
    virtual ~LayerInterface() = default;
    // Return ownership of a heap-allocated TensorInterface so different concrete Tensor3D sizes can be returned.
    virtual std::unique_ptr<TensorInterface> forward(const TensorInterface& input) = 0;
    virtual std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) = 0;
    virtual std::string to_string(bool details = false) const = 0;
    virtual LayerInterface* clone() const = 0;
};

#endif // LAYER_INTERFACE_HPP
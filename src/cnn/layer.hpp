#ifndef LAYER_HPP
#define LAYER_HPP

#include "layerinterface.hpp"
#include "tensorinterface.hpp"
#include "optimizer.hpp"
#include <memory>

template<size_t X_in, size_t Y_in, size_t Z_in, size_t X_out, size_t Y_out, size_t Z_out>
class Layer : public LayerInterface {
public:
    explicit Layer(Optimizer& optimizer) : LayerInterface(optimizer) {}
    virtual ~Layer() override = default;
    virtual std::unique_ptr<TensorInterface> forward(const TensorInterface& input) = 0;
    virtual std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) = 0;
    virtual void update(const Optimizer& optimizer) = 0;
    virtual std::string to_string(bool details = false) const = 0;
    virtual LayerInterface* clone() const = 0;
};

#endif // LAYER_HPP
#ifndef LAYER_INTERFACE_HPP
#define LAYER_INTERFACE_HPP

#include "tensorinterface.hpp"
#include "optimizer.hpp"
#include <memory>

template<template<auto> class Opt, typename... Args>
class LayerInterface {
public:
    virtual ~LayerInterface() = default;
    virtual std::unique_ptr<TensorInterface> forward(const TensorInterface& input) = 0;
    virtual std::unique_ptr<TensorInterface> backward(const TensorInterface& grad_output) = 0;
    virtual void update() = 0;
    virtual std::string to_string(bool details = false) const = 0;
    virtual LayerInterface<Opt, Args...>* clone() const = 0;
};

#endif // LAYER_INTERFACE_HPP
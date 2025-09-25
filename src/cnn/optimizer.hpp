#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "layerinterface.hpp"

class Optimizer {
public:
    ~Optimizer() = default;
    virtual void update(LayerInterface& layer) = 0;
};

#endif // OPTIMIZER_HPP
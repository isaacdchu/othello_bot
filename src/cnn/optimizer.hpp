#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

class Optimizer {
public:
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};

#endif // OPTIMIZER_HPP
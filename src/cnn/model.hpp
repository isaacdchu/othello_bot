#ifndef MODEL_HPP
#define MODEL_HPP

#include "tensor3d.hpp"

class Model {
public:
    virtual ~Model() = default;
    virtual Tensor3D<1, 1, 1> forward(const Tensor3D<8, 8, 3> &input) = 0;
    virtual Tensor3D<8, 8, 3> backward(const Tensor3D<1, 1, 1> &grad_output) = 0;
    virtual void update() = 0;
    virtual void save(const std::string &filepath) const = 0;
    static Model load(const std::string &filepath);
};

#endif // MODEL_HPP
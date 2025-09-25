#ifndef LAYER_HPP
#define LAYER_HPP

#include "layerinterface.hpp"
#include "tensor3d.hpp"

template<size_t X_in, size_t Y_in, size_t Z_in, size_t X_out, size_t Y_out, size_t Z_out>
class Layer : public LayerInterface {
public:
    // Returns the output tensor after applying the layer transformation
    virtual Tensor3D<X_out, Y_out, Z_out> forward(const Tensor3D<X_in, Y_in, Z_in>& input) = 0;
    // Returns the gradient with respect to the input tensor given the gradient of the output
    virtual Tensor3D<X_in, Y_in, Z_in> backward(const Tensor3D<X_out, Y_out, Z_out>& grad_output) = 0;
};

#endif // LAYER_HPP
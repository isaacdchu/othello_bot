#ifndef LAYER_INTERFACE_HPP
#define LAYER_INTERFACE_HPP

class LayerInterface {
public:
    virtual ~LayerInterface() = default;
    virtual void forward() = 0;
    virtual void backward() = 0;
};

#endif // LAYER_INTERFACE_HPP
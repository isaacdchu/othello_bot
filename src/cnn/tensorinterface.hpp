#ifndef TENSORINTERFACE_HPP
#define TENSORINTERFACE_HPP

#include <string>
#include <array>

class TensorInterface {
public:
    virtual ~TensorInterface() = default;
    virtual const float& at(size_t x, size_t y, size_t c) const = 0;
    virtual float& at(size_t x, size_t y, size_t c) = 0;
    virtual const float& at(size_t i) const = 0;
    virtual float& at(size_t i) = 0;
    virtual std::string to_string() const = 0;
    virtual constexpr size_t size() const = 0;
};
#endif // TENSORINTERFACE_HPP
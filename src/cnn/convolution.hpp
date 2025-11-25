#include "tensor3d.hpp"

#include <algorithm>
#include <vector>

template<size_t X_in, size_t Y_in, size_t C_in, size_t K, size_t S, size_t P, size_t C_out>
Tensor3D<(X_in - K + 2 * P) / S + 1, (Y_in - K + 2 * P) / S + 1, C_out> \
convolute(
    const TensorInterface& input,
    const Tensor3D<K, K, C_in * C_out>& weights) {
    // cast to concrete tensor implementation (assumes caller passes Tensor3D)
    const auto& in = static_cast<const Tensor3D<X_in, Y_in, C_in>&>(input);
    constexpr size_t X_out = (X_in - K + 2 * P) / S + 1;
    constexpr size_t Y_out = (Y_in - K + 2 * P) / S + 1;                    
    Tensor3D<X_out, Y_out, C_out> output;

    // im2col: each column is one receptive field (K*K*C_in), columns = X_out * Y_out
    constexpr size_t patch_size = K * K * C_in;
    const size_t n_patches = X_out * Y_out;
    std::vector<float> col(patch_size * n_patches);
    size_t patch_idx = 0;
    for (size_t y = 0; y < Y_out; ++y) {
        for (size_t x = 0; x < X_out; ++x, ++patch_idx) {
            // build single column for position (x,y)
            for (size_t c_in = 0; c_in < C_in; ++c_in) {
                for (size_t ky = 0; ky < K; ++ky) {
                    for (size_t kx = 0; kx < K; ++kx) {
                        int in_x = int(x) * int(S) + int(kx) - int(P);
                        int in_y = int(y) * int(S) + int(ky) - int(P);
                        float val = 0.0f;
                        if (in_x >= 0 && in_y >= 0 &&
                            (unsigned)in_x < X_in && (unsigned)in_y < Y_in) {
                            val = in.at((size_t)in_x, (size_t)in_y, c_in);
                        }
                        size_t row = (c_in * K + ky) * K + kx; // 0..patch_size-1
                        col[row * n_patches + patch_idx] = val;
                    }
                }
            }
        }
    }

    // reshape weights into matrix W (C_out x patch_size)
    std::vector<float> W(C_out * patch_size);
    for (size_t c_out = 0; c_out < C_out; ++c_out) {
        for (size_t c_in = 0; c_in < C_in; ++c_in) {
            for (size_t ky = 0; ky < K; ++ky) {
                for (size_t kx = 0; kx < K; ++kx) {
                    size_t row = (c_in * K + ky) * K + kx;
                    W[c_out * patch_size + row] = weights.at(kx, ky, c_in + c_out * C_in);
                }
            }
        }
    }

    // simple GEMM: out = W * col  (C_out x n_patches)
    std::vector<float> outmat(C_out * n_patches);
    std::fill(outmat.begin(), outmat.end(), 0.0f);
    for (size_t m = 0; m < C_out; ++m) {
        float* outrow = &outmat[m * n_patches];
        const float* wrow = &W[m * patch_size];
        for (size_t k = 0; k < patch_size; ++k) {
            const float w = wrow[k];
            const float* colrow = &col[k * n_patches];
            for (size_t n = 0; n < n_patches; ++n) {
                outrow[n] += w * colrow[n];
            }
        }
    }

    // write back to output tensor and add biases
    for (size_t c_out = 0; c_out < C_out; ++c_out) {
        for (size_t y = 0; y < Y_out; ++y) {
            for (size_t x = 0; x < X_out; ++x) {
                size_t idx = c_out * n_patches + (y * X_out + x);
                output.at(x, y, c_out) = outmat[idx];
            }
        }
    }
    return output;
}

template<size_t X_in, size_t Y_in, size_t C_in, size_t K, size_t S, size_t P, size_t C_out>
Tensor3D<(X_in - K + 2 * P) / S + 1, (Y_in - K + 2 * P) / S + 1, C_out> \
convolute(
    const TensorInterface& input,
    const Tensor3D<K, K, C_in * C_out>& weights,
    const Tensor3D<1, 1, C_out>& biases) {
    constexpr size_t X_out = (X_in - K + 2 * P) / S + 1;
    constexpr size_t Y_out = (Y_in - K + 2 * P) / S + 1;                    
    Tensor3D<X_out, Y_out, C_out> output = convolute<X_in, Y_in, C_in, K, S, P, C_out>(input, weights);

    // add biases to output
    for (size_t c_out = 0; c_out < C_out; ++c_out) {
        for (size_t y = 0; y < Y_out; ++y) {
            for (size_t x = 0; x < X_out; ++x) {
                output.at(x, y, c_out) = output.at(x, y, c_out) + biases.at(0, 0, c_out);
            }
        }
    }

    return output;
}
template <size_t X_in, size_t Y_in, size_t C_in, size_t K, size_t S, size_t P, size_t C_out>
Tensor3D<K, K, C_in * C_out>
grad_convolute(
    const Tensor3D<X_in, Y_in, C_in>& input,
    const Tensor3D<(X_in - K + 2 * P) / S + 1, (Y_in - K + 2 * P) / S + 1, C_out>& delta) {
    Tensor3D<K, K, C_in * C_out> dL_dW;
    dL_dW.fill(0.0f);
    constexpr size_t X_out = (X_in - K + 2 * P) / S + 1;
    constexpr size_t Y_out = (Y_in - K + 2 * P) / S + 1;
    for (size_t c_out = 0; c_out < C_out; ++c_out) {
        for (size_t c_in = 0; c_in < C_in; ++c_in) {
            for (size_t ky = 0; ky < K; ++ky) {
                for (size_t kx = 0; kx < K; ++kx) {
                    float sum = 0.0f;
                    for (size_t y = 0; y < Y_out; ++y) {
                        for (size_t x = 0; x < X_out; ++x) {
                            int in_x = int(x) * int(S) + int(kx) - int(P);
                            int in_y = int(y) * int(S) + int(ky) - int(P);
                            float input_val = 0.0f;
                            if (in_x >= 0 && in_y >= 0 &&
                                (unsigned)in_x < X_in && (unsigned)in_y < Y_in) {
                                input_val = input.at((size_t)in_x, (size_t)in_y, c_in);
                            }
                            sum += input_val * delta.at(x, y, c_out);
                        }
                    }
                    dL_dW.at(kx, ky, c_in + c_out * C_in) = sum;
                }
            }
        }
    }
    return dL_dW;
}

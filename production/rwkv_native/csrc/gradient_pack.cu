// Low-precision GEMM operands only; activation gradients remain FP32.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
using at::Tensor;
namespace {
constexpr int THREADS = 256;
// All values are absolute FP32 magnitudes. Their unsigned encodings preserve
// order, with every NaN above infinity, so NaNs still propagate to the scale.
__device__ float maximum(float a, float b) {
    return __uint_as_float(max(__float_as_uint(a), __float_as_uint(b)));
}
__device__ float block_max(float value) {
    __shared__ unsigned warp[8];
    unsigned bits = __reduce_max_sync(0xffffffff, __float_as_uint(value));
    if (threadIdx.x % 32 == 0)
        warp[threadIdx.x / 32] = bits;
    __syncthreads();
    bits = threadIdx.x < 8 ? warp[threadIdx.x] : 0u;
    if (threadIdx.x < 32)
        bits = __reduce_max_sync(0xffffffff, bits);
    return __uint_as_float(bits);
}
__global__ void partial_max(const float *g, float *partial, int64_t size) {
    float value = 0.f;
    for (int64_t i = int64_t(blockIdx.x) * THREADS + threadIdx.x; i < size;
         i += int64_t(gridDim.x) * THREADS)
        value = maximum(value, fabsf(g[i]));
    value = block_max(value);
    if (!threadIdx.x)
        partial[blockIdx.x] = value;
}
template <bool ABSOLUTE, int BOUND = 0>
__global__ void make_scale(const float *partial, float *scale, float *coefficients, int count) {
    float value = 0.f;
#pragma unroll
    for (int i = threadIdx.x; i < (BOUND ? BOUND : count); i += THREADS)
        if (i < count)
            value = maximum(value, ABSOLUTE ? fabsf(partial[i]) : partial[i]);
    value = block_max(value);
    if (!threadIdx.x) {
        const unsigned bits = __float_as_uint(value) & 0x7fffffffu;
        const unsigned field = bits >> 23;
        // frexp exponent, including zero, subnormal and nonfinite inputs.
        int exponent = (field && field < 255) ? int(field) - 126
                                              : (!field && bits ? 31 - __clz(bits) - 148 : 0);
        const int shift = max(-120, min(120, 7 - exponent));
        scale[0] = __uint_as_float(unsigned(127 + shift) << 23);
        coefficients[0] = __uint_as_float(unsigned(127 - shift) << 23);
        coefficients[1] = 0.f;
    }
}
// Block zero reduces and packs small gradients while the other blocks copy x.
// These tasks are independent, so they share one launch without a grid barrier.
template <class scalar_t, int PARTS, int BOUND>
__global__ void small_pack(const float *g, const scalar_t *x, float *scale, float *coefficients,
                           scalar_t *packed, scalar_t *repeated, int64_t ng, int64_t nx,
                           bool vector_x) {
    if (blockIdx.x) {
        if (vector_x) {
            for (int64_t i = int64_t(blockIdx.x - 1) * THREADS + threadIdx.x; i < nx / 8;
                 i += int64_t(gridDim.x - 1) * THREADS) {
                const uint4 value = reinterpret_cast<const uint4 *>(x)[i];
#pragma unroll
                for (int p = 0; p < PARTS; ++p)
                    reinterpret_cast<uint4 *>(repeated)[p * (nx / 8) + i] = value;
            }
        } else {
            for (int64_t i = int64_t(blockIdx.x - 1) * THREADS + threadIdx.x; i < nx;
                 i += int64_t(gridDim.x - 1) * THREADS) {
                const scalar_t value = x[i];
#pragma unroll
                for (int p = 0; p < PARTS; ++p)
                    repeated[p * nx + i] = value;
            }
        }
        return;
    }
    float values[BOUND / THREADS], value = 0.f;
#pragma unroll
    for (int q = 0; q < BOUND / THREADS; ++q) {
        int i = threadIdx.x + q * THREADS;
        values[q] = i < ng ? g[i] : 0.f;
        value = maximum(value, fabsf(values[q]));
    }
    value = block_max(value);
    __shared__ float factor;
    if (!threadIdx.x) {
        unsigned bits = __float_as_uint(value) & 0x7fffffffu, field = bits >> 23;
        int exponent = (field && field < 255) ? int(field) - 126
                                              : (!field && bits ? 31 - __clz(bits) - 148 : 0);
        int shift = max(-120, min(120, 7 - exponent));
        factor = __uint_as_float(unsigned(127 + shift) << 23);
        scale[0] = factor;
        coefficients[0] = __uint_as_float(unsigned(127 - shift) << 23);
        coefficients[1] = 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < BOUND / THREADS; ++q) {
        int i = threadIdx.x + q * THREADS;
        if (i < ng) {
            const float scaled = __fmul_rn(values[q], factor);
            const scalar_t high = scalar_t(scaled);
            const float rest = __fsub_rn(scaled, float(high));
            const scalar_t low = scalar_t(rest);
            packed[i] = high;
            packed[ng + i] = low;
            if constexpr (PARTS == 3)
                packed[2 * ng + i] = scalar_t(__fsub_rn(rest, float(low)));
        }
    }
}
template <class scalar_t, int PARTS>
__global__ void pack_operands(const float *g, const scalar_t *x, const float *scale,
                              scalar_t *packed, scalar_t *repeated, int64_t ng, int64_t nx,
                              bool vector_x) {
    for (int64_t i = int64_t(blockIdx.x) * THREADS + threadIdx.x;
         i < max(ng, vector_x ? nx / 8 : nx); i += int64_t(gridDim.x) * THREADS) {
        if (i < ng) {
            const float scaled = __fmul_rn(g[i], scale[0]);
            const scalar_t high = scalar_t(scaled);
            const float rest = __fsub_rn(scaled, float(high));
            const scalar_t low = scalar_t(rest);
            packed[i] = high;
            packed[ng + i] = low;
            if constexpr (PARTS == 3)
                packed[2 * ng + i] = scalar_t(__fsub_rn(rest, float(low)));
        }
        if (vector_x) {
            if (i < nx / 8) {
                const uint4 value = reinterpret_cast<const uint4 *>(x)[i];
#pragma unroll
                for (int p = 0; p < PARTS; ++p)
                    reinterpret_cast<uint4 *>(repeated)[p * (nx / 8) + i] = value;
            }
        } else if (i < nx) {
            const scalar_t value = x[i];
            for (int p = 0; p < PARTS; ++p)
                repeated[p * nx + i] = value;
        }
    }
}
template <int PARTS>
__global__ void combine_parts(const float *products, const float *scale, float *out, int64_t size) {
    // scale is a normal power of two with exponent in [-120,120].
    const float inverse = __frcp_rn(scale[0]);
    for (int64_t i = int64_t(blockIdx.x) * THREADS + threadIdx.x; i < size;
         i += int64_t(gridDim.x) * THREADS) {
        float correction = products[size + i];
        if constexpr (PARTS == 3)
            correction = __fadd_rn(correction, products[2 * size + i]);
        out[i] = __fmul_rn(__fadd_rn(products[i], correction), inverse);
    }
}
} // namespace
std::vector<Tensor> pack(Tensor x, Tensor g, bool need_dw) {
    TORCH_CHECK(x.is_cuda() && g.device() == x.device() && x.dim() == 2 && g.dim() == 2 &&
                    x.size(0) == g.size(0),
                "expected matching CUDA matrices");
    TORCH_CHECK(x.is_contiguous() && g.is_contiguous() && g.scalar_type() == at::kFloat &&
                    g.numel() > 0,
                "expected contiguous operands and nonempty FP32 gradient");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "expected FP16/BF16 operands");
    c10::cuda::CUDAGuard guard(x.device());
    int parts = x.scalar_type() == at::kHalf ? 2 : 3;
    int blocks = std::min<int64_t>(1024, (g.numel() + THREADS - 1) / THREADS);
    auto partial = at::empty({blocks}, g.options()), scale = at::empty({}, g.options()),
         coefficients = at::empty({2}, g.options());
    auto packed = at::empty({g.size(0) * parts, g.size(1)}, x.options());
    auto repeated = at::empty({need_dw ? x.size(0) * parts : 0, x.size(1)}, x.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    int64_t nx = need_dw ? x.numel() : 0;
    bool vector_x = nx % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
    if (g.numel() <= 16384) {
        int copy_blocks =
            std::min<int64_t>(4096, ((vector_x ? nx / 8 : nx) + THREADS - 1) / THREADS);
#define SMALL(TYPE, PARTS, BOUND)                                                                  \
    small_pack<TYPE, PARTS, BOUND><<<1 + copy_blocks, THREADS, 0, stream>>>(                       \
        g.data_ptr<float>(), x.data_ptr<TYPE>(), scale.data_ptr<float>(),                          \
        coefficients.data_ptr<float>(), packed.data_ptr<TYPE>(), repeated.data_ptr<TYPE>(),        \
        g.numel(), nx, vector_x)
        if (parts == 2) {
            if (g.numel() <= 4096) {
                SMALL(at::Half, 2, 4096);
            } else if (g.numel() <= 8192) {
                SMALL(at::Half, 2, 8192);
            } else {
                SMALL(at::Half, 2, 16384);
            }
        } else {
            if (g.numel() <= 4096) {
                SMALL(at::BFloat16, 3, 4096);
            } else if (g.numel() <= 8192) {
                SMALL(at::BFloat16, 3, 8192);
            } else {
                SMALL(at::BFloat16, 3, 16384);
            }
        }
#undef SMALL
    } else {
        partial_max<<<blocks, THREADS, 0, stream>>>(g.data_ptr<float>(), partial.data_ptr<float>(),
                                                    g.numel());
        make_scale<false><<<1, THREADS, 0, stream>>>(partial.data_ptr<float>(),
                                                     scale.data_ptr<float>(),
                                                     coefficients.data_ptr<float>(), blocks);
        blocks = std::min<int64_t>(
            4096, (std::max(g.numel(), vector_x ? nx / 8 : nx) + THREADS - 1) / THREADS);
        if (parts == 2)
            pack_operands<at::Half, 2><<<blocks, THREADS, 0, stream>>>(
                g.data_ptr<float>(), x.data_ptr<at::Half>(), scale.data_ptr<float>(),
                packed.data_ptr<at::Half>(), repeated.data_ptr<at::Half>(), g.numel(), nx,
                vector_x);
        else
            pack_operands<at::BFloat16, 3><<<blocks, THREADS, 0, stream>>>(
                g.data_ptr<float>(), x.data_ptr<at::BFloat16>(), scale.data_ptr<float>(),
                packed.data_ptr<at::BFloat16>(), repeated.data_ptr<at::BFloat16>(), g.numel(), nx,
                vector_x);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed, repeated, coefficients, scale};
}
Tensor combine(Tensor products, Tensor scale, int64_t parts) {
    TORCH_CHECK(products.is_cuda() && products.dim() == 2 && products.scalar_type() == at::kFloat &&
                    products.is_contiguous(),
                "expected contiguous FP32 products");
    TORCH_CHECK((parts == 2 || parts == 3) && products.size(0) % parts == 0,
                "invalid component count");
    TORCH_CHECK(scale.device() == products.device() && scale.scalar_type() == at::kFloat &&
                    scale.numel() == 1 && scale.is_contiguous(),
                "invalid scale");
    c10::cuda::CUDAGuard guard(products.device());
    auto out = at::empty({products.size(0) / parts, products.size(1)}, products.options());
    if (out.numel()) {
        int blocks = std::min<int64_t>(4096, (out.numel() + THREADS - 1) / THREADS);
        auto stream = c10::cuda::getCurrentCUDAStream();
        if (parts == 2)
            combine_parts<2><<<blocks, THREADS, 0, stream>>>(products.data_ptr<float>(),
                                                             scale.data_ptr<float>(),
                                                             out.data_ptr<float>(), out.numel());
        else
            combine_parts<3><<<blocks, THREADS, 0, stream>>>(products.data_ptr<float>(),
                                                             scale.data_ptr<float>(),
                                                             out.data_ptr<float>(), out.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack", &pack);
    m.def("combine", &combine);
}

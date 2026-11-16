// Fixed per-row FP32 normalization, independent of token count or batch order.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
using at::Tensor;
template <typename P>
__global__ void value_blend(const float *value, const float *first, const P *bias, const float *raw,
                            float *out, int64_t count, int C) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float gate = 1.f / (1.f + expf(-__fadd_rn(raw[i], float(bias[i % C]))));
        out[i] = __fadd_rn(value[i], __fmul_rn(first[i] - value[i], gate));
    }
}
Tensor fixed_value_blend(Tensor value, Tensor first, Tensor bias, Tensor raw) {
    TORCH_CHECK(value.is_cuda() && value.dim() >= 2 && value.size(-1) > 0,
                "expected CUDA activations with positive channel count");
    for (auto x : {value, first, raw})
        TORCH_CHECK(x.device() == value.device() && x.scalar_type() == at::kFloat &&
                        x.sizes() == value.sizes() && x.is_contiguous(),
                    "expected same-shaped contiguous FP32 activations");
    TORCH_CHECK(bias.device() == value.device() && bias.dim() == 1 &&
                    bias.numel() == value.size(-1) && bias.is_contiguous(),
                "invalid bias");
    TORCH_CHECK(bias.scalar_type() == at::kHalf || bias.scalar_type() == at::kBFloat16 ||
                    bias.scalar_type() == at::kFloat,
                "invalid bias dtype");
    c10::cuda::CUDAGuard guard(value.device());
    auto out = at::empty_like(value);
    if (value.numel()) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, bias.scalar_type(), "fixed_value_blend", [&] {
                value_blend<scalar_t>
                    <<<(value.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                        value.data_ptr<float>(), first.data_ptr<float>(), bias.data_ptr<scalar_t>(),
                        raw.data_ptr<float>(), out.data_ptr<float>(), value.numel(),
                        value.size(-1));
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
template <typename P> __global__ void relu_pack(const float *x, P *out, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = fmaxf(x[i], 0.f);
        out[i] = P(value * value);
    }
}
Tensor fixed_relu_pack(Tensor x, bool bf16) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
                "expected contiguous FP32 activations");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x, x.options().dtype(bf16 ? at::kBFloat16 : at::kHalf));
    if (x.numel()) {
        if (bf16)
            relu_pack<at::BFloat16>
                <<<(x.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(), out.data_ptr<at::BFloat16>(), x.numel());
        else
            relu_pack<at::Half>
                <<<(x.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(), out.data_ptr<at::Half>(), x.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
__device__ float norm_sum(float value, float *scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int d = 16; d; d >>= 1)
        value += __shfl_down_sync(0xffffffff, value, d);
    if (!lane)
        scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? scratch[threadIdx.x] : 0.f;
    if (!warp) {
        for (int d = 16; d; d >>= 1)
            value += __shfl_down_sync(0xffffffff, value, d);
        if (!lane)
            scratch[8] = value;
    }
    __syncthreads();
    return scratch[8];
}
template <typename P>
__global__ __launch_bounds__(256, 2) void fixed_norm(const float *x, const P *w, const P *b,
                                                     float *out, int C) {
    __shared__ float scratch[18];
    const int64_t base = int64_t(blockIdx.x) * C;
    float sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256)
        sum += x[base + c];
    const float mean = norm_sum(sum, scratch) / C;
    sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256) {
        float d = x[base + c] - mean;
        sum = fmaf(d, d, sum);
    }
    const float inverse = rsqrtf(norm_sum(sum, scratch + 9) / C + 1e-5f);
    for (int c = threadIdx.x; c < C; c += 256)
        out[base + c] =
            __fadd_rn(__fmul_rn((x[base + c] - mean) * inverse, float(w[c])), float(b[c]));
}
Tensor fixed_layer_norm(Tensor x, Tensor w, Tensor b) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 2 && x.size(-1) > 0 &&
                    x.scalar_type() == at::kFloat,
                "expected contiguous FP32 activations");
    TORCH_CHECK(w.device() == x.device() && b.device() == x.device() && w.dim() == 1 &&
                    b.sizes() == w.sizes() && w.numel() == x.size(-1) && w.is_contiguous() &&
                    b.is_contiguous() && b.scalar_type() == w.scalar_type(),
                "invalid normalization parameters");
    TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kBFloat16 ||
                    w.scalar_type() == at::kFloat,
                "invalid parameter dtype");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    int C = x.size(-1);
    int64_t rows = x.numel() / C;
    if (rows) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, w.scalar_type(), "fixed_layer_norm", [&] {
                fixed_norm<scalar_t><<<rows, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(), w.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                    out.data_ptr<float>(), C);
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

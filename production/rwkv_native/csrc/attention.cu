// FP32 residual/norm/mixing and state; narrow only the actual GEMM operand.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
using at::Tensor;

__device__ float frontend_sum(float value, float *scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
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

// Attention normalization and six mixing branches share the same FP32 z.
template <typename P> struct AttentionCoefficients {
    const P *values[6];
};

// Same normalization/mixing boundary as Albatross ln_mix6: each normalized
// value is shared by six real GEMM operands. Prefill reads the previous row
// directly; no materialized shifted tensor or FP32 branch tensors are needed.
template <typename P>
__global__ __launch_bounds__(256,
                             2) void attention_prefill_kernel(const float *z, const float *previous,
                                                              AttentionCoefficients<P> coefficients,
                                                              P *mixed, float *state, int T, int C,
                                                              int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < count) {
        const int c = i % C;
        const int64_t row = i / C;
        const int t = row % T, b = row / T;
        const float current = z[i], past = t ? z[i - C] : previous[int64_t(b) * C + c];
        const float delta = past - current;
#pragma unroll
        for (int g = 0; g < 6; ++g)
            mixed[int64_t(g) * count + i] =
                P(__fadd_rn(current, __fmul_rn(delta, float(coefficients.values[g][c]))));
        if (t == T - 1)
            state[int64_t(b) * C + c] = current;
    }
}

std::vector<Tensor> packed_attention_prefill(Tensor z, Tensor previous,
                                             std::vector<Tensor> coefficients) {
    TORCH_CHECK(z.is_cuda() && z.dim() == 3 && z.size(1) > 0 && z.size(2) > 0 &&
                    z.is_contiguous() && z.scalar_type() == at::kFloat,
                "expected contiguous FP32 [B,T,C]");
    const int B = z.size(0), T = z.size(1), C = z.size(2);
    TORCH_CHECK(previous.device() == z.device() && previous.scalar_type() == at::kFloat &&
                    previous.is_contiguous() && previous.sizes() == at::IntArrayRef({B, C}),
                "expected FP32 [B,C] state");
    TORCH_CHECK(coefficients.size() == 6, "expected six coefficients");
    const auto dtype = coefficients[0].scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "expected FP16/BF16 coefficients");
    for (const auto &c : coefficients)
        TORCH_CHECK(c.device() == z.device() && c.scalar_type() == dtype && c.is_contiguous() &&
                        c.dim() == 1 && c.numel() == C,
                    "invalid mixing coefficient");
    c10::cuda::CUDAGuard guard(z.device());
    auto mixed = at::empty({6, B, T, C}, coefficients[0].options()),
         state = at::empty_like(previous);
    if (z.numel()) {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "attention_prefill", [&] {
            AttentionCoefficients<scalar_t> pointers{};
            for (int g = 0; g < 6; ++g)
                pointers.values[g] = coefficients[g].data_ptr<scalar_t>();
            attention_prefill_kernel<scalar_t>
                <<<(z.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    z.data_ptr<float>(), previous.data_ptr<float>(), pointers,
                    mixed.data_ptr<scalar_t>(), state.data_ptr<float>(), T, C, z.numel());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {mixed, state};
}
template <typename P, bool SPLIT>
__global__ __launch_bounds__(256, 2) void attention_frontend_kernel(
    const float *x, const float *previous, const P *weight, const P *bias,
    AttentionCoefficients<P> coefficients, const bool *mask, P *mixed, float *normalized,
    float *state, int B, int C) {
    const int row = blockIdx.x;
    const int64_t base = int64_t(row) * C, span = int64_t(B) * C;
    __shared__ float scratch[18];
    float sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256)
        sum += x[base + c];
    const float mean = frontend_sum(sum, scratch) / C;
    sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256) {
        const float d = x[base + c] - mean;
        sum = fmaf(d, d, sum);
    }
    const float inverse = rsqrtf(frontend_sum(sum, scratch + 9) / C + 1e-5f);
    for (int c = threadIdx.x; c < C; c += 256) {
        const int64_t i = base + c;
        const float z =
            __fadd_rn(__fmul_rn((x[i] - mean) * inverse, float(weight[c])), float(bias[c]));
        const float delta = previous[i] - z;
        if constexpr (SPLIT) {
            const int g = blockIdx.y;
            mixed[g * span + i] =
                P(__fadd_rn(z, __fmul_rn(delta, float(coefficients.values[g][c]))));
        } else {
#pragma unroll
            for (int g = 0; g < 6; ++g)
                mixed[g * span + i] =
                    P(__fadd_rn(z, __fmul_rn(delta, float(coefficients.values[g][c]))));
        }
        if (!SPLIT || blockIdx.y == 0) {
            if (normalized)
                normalized[i] = z;
            state[i] = mask[row] ? z : previous[i];
        }
    }
}

std::vector<Tensor> packed_attention_frontend(std::vector<Tensor> xs, Tensor mask, bool save,
                                              bool split) {
    TORCH_CHECK(xs.size() == 10,
                "expected x, previous state, norm parameters and six coefficients");
    const auto x = xs[0];
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.size(1) == 1 && x.size(2) > 0 &&
                    x.is_contiguous() && x.scalar_type() == at::kFloat,
                "expected FP32 [B,1,C]");
    const int B = x.size(0), C = x.size(2);
    const auto dtype = xs[2].scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "expected FP16/BF16 parameters");
    for (int i = 0; i < 10; ++i) {
        TORCH_CHECK(xs[i].device() == x.device() && xs[i].is_contiguous(),
                    "device/layout mismatch");
        if (i == 1)
            TORCH_CHECK(xs[i].sizes() == at::IntArrayRef({B, C}) &&
                            xs[i].scalar_type() == at::kFloat,
                        "expected FP32 [B,C]");
        if (i >= 2)
            TORCH_CHECK(xs[i].dim() == 1 && xs[i].numel() == C && xs[i].scalar_type() == dtype,
                        "invalid parameter");
    }
    TORCH_CHECK(mask.device() == x.device() && mask.is_contiguous() &&
                    mask.scalar_type() == at::kBool && mask.sizes() == at::IntArrayRef({B, 1}),
                "expected bool [B,1] mask");
    c10::cuda::CUDAGuard guard(x.device());
    auto mixed = at::empty({6, B, 1, C}, xs[2].options()),
         normalized = save ? at::empty_like(x) : at::empty({0}, x.options());
    auto state = at::empty_like(xs[1]);
    if (B) {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "fixed_attention_frontend", [&] {
            AttentionCoefficients<scalar_t> coefficients{};
            for (int g = 0; g < 6; ++g)
                coefficients.values[g] = xs[g + 4].data_ptr<scalar_t>();
            if (split) {
                attention_frontend_kernel<scalar_t, true>
                    <<<dim3(B, 6), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                        x.data_ptr<float>(), xs[1].data_ptr<float>(), xs[2].data_ptr<scalar_t>(),
                        xs[3].data_ptr<scalar_t>(), coefficients, mask.data_ptr<bool>(),
                        mixed.data_ptr<scalar_t>(), save ? normalized.data_ptr<float>() : nullptr,
                        state.data_ptr<float>(), B, C);
            } else {
                attention_frontend_kernel<scalar_t, false>
                    <<<B, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                        x.data_ptr<float>(), xs[1].data_ptr<float>(), xs[2].data_ptr<scalar_t>(),
                        xs[3].data_ptr<scalar_t>(), coefficients, mask.data_ptr<bool>(),
                        mixed.data_ptr<scalar_t>(), save ? normalized.data_ptr<float>() : nullptr,
                        state.data_ptr<float>(), B, C);
            }
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {mixed, normalized, state};
}

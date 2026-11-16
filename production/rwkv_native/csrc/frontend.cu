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

template <typename P, bool PREFILL = false>
__global__ __launch_bounds__(256, 2) void frontend_kernel(const float *x, const float *product,
                                                          const float *previous, const P *weight,
                                                          const P *bias, const P *coefficient,
                                                          const bool *mask, P *packed,
                                                          float *normalized, float *state,
                                                          float *residual, int C) {
    const int row = blockIdx.x;
    const int64_t base = int64_t(row) * C;
    __shared__ float scratch[18];
    float sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256)
        sum += __fadd_rn(x[base + c], product[base + c]);
    const float mean = frontend_sum(sum, scratch) / C;
    sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256) {
        const float d = __fadd_rn(x[base + c], product[base + c]) - mean;
        sum = fmaf(d, d, sum);
    }
    const float inverse = rsqrtf(frontend_sum(sum, scratch + 9) / C + 1e-5f);
    for (int c = threadIdx.x; c < C; c += 256) {
        const int64_t i = base + c;
        const float value = __fadd_rn(x[i], product[i]);
        const float z =
            __fadd_rn(__fmul_rn((value - mean) * inverse, float(weight[c])), float(bias[c]));
        if constexpr (!PREFILL) {
            packed[i] = P(__fadd_rn(z, __fmul_rn(previous[i] - z, float(coefficient[c]))));
            state[i] = mask[row] ? z : previous[i];
        }
        if (normalized)
            normalized[i] = z;
        residual[i] = value;
    }
}

// Albatross ln_cmix boundary, extended to prefill without a shifted tensor.
template <typename P>
__global__
__launch_bounds__(256, 2) void frontend_mix_prefill(const float *z, const float *previous,
                                                    const P *coefficient, P *packed, float *state,
                                                    int T, int C, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < count) {
        const int c = i % C;
        const int64_t row = i / C;
        const int t = row % T, b = row / T;
        const float current = z[i], past = t ? z[i - C] : previous[int64_t(b) * C + c];
        packed[i] = P(__fadd_rn(current, __fmul_rn(past - current, float(coefficient[c]))));
        if (t == T - 1)
            state[int64_t(b) * C + c] = current;
    }
}

std::vector<Tensor> fixed_frontend(std::vector<Tensor> xs, Tensor mask, bool save) {
    TORCH_CHECK(xs.size() == 6,
                "expected residual, projection, previous state and three parameters");
    const auto x = xs[0];
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.size(1) > 0 && x.size(2) > 0 &&
                    x.is_contiguous() && x.scalar_type() == at::kFloat,
                "expected FP32 [B,T,C]");
    const int B = x.size(0), T = x.size(1), C = x.size(2);
    const auto dtype = xs[3].scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "expected FP16/BF16 parameters");
    for (int i = 0; i < 6; ++i) {
        TORCH_CHECK(xs[i].device() == x.device() && xs[i].is_contiguous(),
                    "device/layout mismatch");
        if (i == 1)
            TORCH_CHECK(xs[i].sizes() == at::IntArrayRef({B * T, C}) &&
                            xs[i].scalar_type() == at::kFloat,
                        "expected FP32 [B*T,C] projection");
        if (i == 2)
            TORCH_CHECK(xs[i].sizes() == at::IntArrayRef({B, C}) &&
                            xs[i].scalar_type() == at::kFloat,
                        "expected FP32 [B,C] state");
        if (i >= 3)
            TORCH_CHECK(xs[i].dim() == 1 && xs[i].numel() == C && xs[i].scalar_type() == dtype,
                        "invalid parameter");
    }
    TORCH_CHECK(mask.device() == x.device() && mask.is_contiguous() &&
                    mask.scalar_type() == at::kBool && mask.sizes() == at::IntArrayRef({B, T}),
                "expected bool [B,T] mask");
    c10::cuda::CUDAGuard guard(x.device());
    auto packed = at::empty({B * T, C}, xs[3].options()),
         normalized = (save || T > 1) ? at::empty_like(x) : at::empty({0}, x.options());
    auto state = at::empty_like(xs[2]), residual = at::empty_like(x);
    if (B) {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "fixed_frontend", [&] {
            if (T == 1)
                frontend_kernel<scalar_t><<<B, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(), xs[1].data_ptr<float>(), xs[2].data_ptr<float>(),
                    xs[3].data_ptr<scalar_t>(), xs[4].data_ptr<scalar_t>(),
                    xs[5].data_ptr<scalar_t>(), mask.data_ptr<bool>(), packed.data_ptr<scalar_t>(),
                    save ? normalized.data_ptr<float>() : nullptr, state.data_ptr<float>(),
                    residual.data_ptr<float>(), C);
            else {
                frontend_kernel<scalar_t, true>
                    <<<B * T, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                        x.data_ptr<float>(), xs[1].data_ptr<float>(), xs[2].data_ptr<float>(),
                        xs[3].data_ptr<scalar_t>(), xs[4].data_ptr<scalar_t>(),
                        xs[5].data_ptr<scalar_t>(), mask.data_ptr<bool>(),
                        packed.data_ptr<scalar_t>(), normalized.data_ptr<float>(),
                        state.data_ptr<float>(), residual.data_ptr<float>(), C);
                frontend_mix_prefill<scalar_t>
                    <<<(x.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                        normalized.data_ptr<float>(), xs[2].data_ptr<float>(),
                        xs[5].data_ptr<scalar_t>(), packed.data_ptr<scalar_t>(),
                        state.data_ptr<float>(), T, C, x.numel());
            }
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {packed, save ? normalized : at::empty({0}, x.options()), state, residual};
}

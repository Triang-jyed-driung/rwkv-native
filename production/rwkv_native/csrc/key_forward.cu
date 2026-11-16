// FP32 normalized-key and adaptation gates; mixed-precision parameters are
// expanded in registers. One warp owns each 64-element head.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
using at::Tensor;

template <typename P>
__global__ __launch_bounds__(128, 2) void key_forward_rows(const float *k, const P *scale,
                                                           const P *bias, const float *raw,
                                                           const P *adapt, float *out, float *a,
                                                           float *b, int64_t heads, int H) {
    const int lane = threadIdx.x & 31;
    const int64_t head = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (head >= heads)
        return;
    const int column = (head % H) * 64;
    float key[2], u[2], rate[2], sum = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c = column + lane + 32 * j;
        const int64_t i = head * 64 + lane + 32 * j;
        key[j] = k[i];
        u[j] = key[j] * float(scale[c]);
        rate[j] = 1.f / (1.f + expf(-(raw[i] + float(bias[c]))));
        sum += u[j] * u[j];
    }
    for (int d = 16; d; d >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, d);
    const float inverse = 1.f / fmaxf(sqrtf(__shfl_sync(0xffffffff, sum, 0)), 1e-12f);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c = column + lane + 32 * j;
        const int64_t i = head * 64 + lane + 32 * j;
        const float q = u[j] * inverse;
        out[i] = key[j] * (1.f + (rate[j] - 1.f) * float(adapt[c]));
        a[i] = -q;
        b[i] = q * rate[j];
    }
}

std::vector<Tensor> key_gate_forward(std::vector<Tensor> xs, int64_t H) {
    TORCH_CHECK(xs.size() == 5, "expected key, scale, bias, raw gate, adaptation");
    const auto k = xs[0];
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.dim() >= 2 && k.scalar_type() == at::kFloat &&
                    H > 0 && k.size(-1) == H * 64,
                "invalid FP32 key input");
    c10::cuda::CUDAGuard guard(k.device());
    TORCH_CHECK(xs[3].sizes() == k.sizes() && xs[3].scalar_type() == at::kFloat,
                "invalid raw gate");
    const auto dtype = xs[1].scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
                "invalid parameter dtype");
    for (int i = 0; i < 5; ++i) {
        TORCH_CHECK(xs[i].device() == k.device() && xs[i].is_contiguous(),
                    "device/layout mismatch");
        if (i != 0 && i != 3)
            TORCH_CHECK(xs[i].dim() == 1 && xs[i].numel() == H * 64 && xs[i].scalar_type() == dtype,
                        "invalid parameter shape/dtype");
    }
    auto out = at::empty_like(k), a = at::empty_like(k), b = at::empty_like(k);
    const int64_t heads = k.numel() / 64;
    if (heads) {
        auto stream = c10::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dtype, "key_gate_forward", [&] {
            key_forward_rows<scalar_t><<<(heads + 3) / 4, 128, 0, stream>>>(
                k.data_ptr<float>(), xs[1].data_ptr<scalar_t>(), xs[2].data_ptr<scalar_t>(),
                xs[3].data_ptr<float>(), xs[4].data_ptr<scalar_t>(), out.data_ptr<float>(),
                a.data_ptr<float>(), b.data_ptr<float>(), heads, H);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {out, a, b};
}

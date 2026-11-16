// FP32 readout: group normalization, per-head RKV residual, and output gate.
// One warp owns each 64-element head; no batch-dependent reduction order.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
using at::Tensor;
template <typename P> struct ReadoutInput {
    const float *x[8];
    const P *p[3];
};
__device__ float readout_sum(float x) {
    for (int d = 16; d; d >>= 1)
        x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}
template <typename P>
__global__ __launch_bounds__(32, 2) void readout_rows(ReadoutInput<P> a, float *out, int H) {
    const int group = blockIdx.x, head = group % H, lane = threadIdx.x;
    float x[2], dot = 0.f, sum = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int64_t i = int64_t(group) * 64 + lane + j * 32;
        const int c = head * 64 + lane + j * 32;
        x[j] = a.x[0][i];
        sum += x[j];
        dot += a.x[1][i] * a.x[2][i] * float(a.p[0][c]);
    }
    const float mean = readout_sum(sum) / 64.f;
    sum = 0.f;
    for (int j = 0; j < 2; ++j) {
        float d = x[j] - mean;
        sum += d * d;
    }
    const float inv = rsqrtf(readout_sum(sum) / 64.f + 64e-5f);
    dot = readout_sum(dot);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int64_t i = int64_t(group) * 64 + lane + j * 32;
        const int c = head * 64 + lane + j * 32;
        out[i] = ((x[j] - mean) * inv * float(a.p[1][c]) + float(a.p[2][c]) + dot * a.x[3][i]) *
                 a.x[7][i];
    }
}
Tensor readout_forward(std::vector<Tensor> xs, int64_t H) {
    TORCH_CHECK(xs.size() == 8, "expected x,r,k,v,scale,weight,bias,gate");
    const auto x = xs[0];
    TORCH_CHECK(x.dim() >= 2 && H > 0 && x.size(-1) == H * 64, "expected 64-wide heads");
    const auto dtype = xs[4].scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
                "expected FP32/FP16/BF16 parameters");
    for (int i = 0; i < 8; ++i) {
        const auto t = xs[i];
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == x.device(),
                    "expected contiguous inputs on one CUDA device");
        if (i >= 4 && i <= 6) {
            TORCH_CHECK(t.dim() == 1 && t.numel() == H * 64 && t.scalar_type() == dtype,
                        "invalid parameter shape/dtype");
        } else {
            TORCH_CHECK(t.sizes() == x.sizes() && t.scalar_type() == at::kFloat,
                        "expected FP32 activations of identical shape");
        }
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    if (x.numel()) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dtype, "readout_forward", [&] {
            ReadoutInput<scalar_t> args{};
            for (int i = 0; i < 8; ++i) {
                if (i >= 4 && i <= 6)
                    args.p[i - 4] = xs[i].data_ptr<scalar_t>();
                else
                    args.x[i] = xs[i].data_ptr<float>();
            }
            readout_rows<scalar_t><<<x.numel() / 64, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
                args, out.data_ptr<float>(), H);
        });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

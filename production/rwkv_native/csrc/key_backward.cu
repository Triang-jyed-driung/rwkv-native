// Fused normalized-key / adaptation-gate adjoint; FP32 reductions, no atomics.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
using at::Tensor;
__device__ float key_sum(float v) {
    for (int d = 16; d; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    return __shfl_sync(0xffffffff, v, 0);
}
template <typename T> struct KeyArgs {
    const T *x[5];
    const T *g[3];
    T *dx[5];
};
template <typename T>
__global__ __launch_bounds__(32, 2) void key_backward_rows(KeyArgs<T> a, float *partial, int C,
                                                           int H) {
    const int row = blockIdx.x / H, head = blockIdx.x % H, lane = threadIdx.x;
    float k[2], scale[2], u[2], q[2], rate[2], ka[2], gk[2], ga[2], gb[2], gq[2];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c = head * 64 + lane + j * 32;
        const int64_t i = int64_t(row) * C + c;
        k[j] = float(a.x[0][i]);
        scale[j] = float(a.x[1][c]);
        ka[j] = float(a.x[4][c]);
        u[j] = k[j] * scale[j];
        rate[j] = 1.f / (1.f + expf(-(float(a.x[2][c]) + float(a.x[3][i]))));
        gk[j] = a.g[0] ? float(a.g[0][i]) : 0.f;
        ga[j] = a.g[1] ? float(a.g[1][i]) : 0.f;
        gb[j] = a.g[2] ? float(a.g[2][i]) : 0.f;
        sum += u[j] * u[j];
    }
    const float norm = sqrtf(key_sum(sum)), inverse = 1.f / fmaxf(norm, 1e-12f);
    float dot = 0.f;
    for (int j = 0; j < 2; ++j) {
        q[j] = u[j] * inverse;
        gq[j] = -ga[j] + gb[j] * rate[j];
        dot += gq[j] * q[j];
    }
    dot = key_sum(dot);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c = head * 64 + lane + j * 32;
        const int64_t i = int64_t(row) * C + c;
        const float du = (gq[j] - (norm >= 1e-12f ? q[j] * dot : 0.f)) * inverse;
        const float dr = (gb[j] * q[j] + gk[j] * k[j] * ka[j]) * rate[j] * (1.f - rate[j]);
        a.dx[0][i] = T(gk[j] * (1.f + (rate[j] - 1.f) * ka[j]) + du * scale[j]);
        a.dx[3][i] = T(dr);
        partial[(int64_t(row) * 3) * C + c] = du * k[j];
        partial[(int64_t(row) * 3 + 1) * C + c] = dr;
        partial[(int64_t(row) * 3 + 2) * C + c] = gk[j] * k[j] * (rate[j] - 1.f);
    }
}
template <typename T>
__global__ void key_backward_parameters(KeyArgs<T> a, const float *p, int rows, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float ds = 0.f, da = 0.f, dk = 0.f;
    for (int row = 0; row < rows; ++row) {
        ds += p[(int64_t(row) * 3) * C + c];
        da += p[(int64_t(row) * 3 + 1) * C + c];
        dk += p[(int64_t(row) * 3 + 2) * C + c];
    }
    a.dx[1][c] = T(ds);
    a.dx[2][c] = T(da);
    a.dx[4][c] = T(dk);
}
std::vector<Tensor> key_gate_backward(std::vector<Tensor> xs, std::vector<Tensor> grads,
                                      int64_t H) {
    TORCH_CHECK(xs.size() == 5 && grads.size() == 3, "invalid number of inputs");
    auto x = xs[0];
    TORCH_CHECK(x.is_cuda() && x.dim() >= 2 && H > 0 && x.size(-1) == 64 * H, "invalid key shape");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "unsupported dtype");
    const int C = x.size(-1), rows = x.numel() / C;
    auto check = [&](const Tensor &t) {
        TORCH_CHECK(t.is_contiguous() && t.device() == x.device() &&
                        t.scalar_type() == x.scalar_type(),
                    "dtype/device/layout mismatch");
    };
    c10::cuda::CUDAGuard guard(x.device());
    std::vector<Tensor> dx;
    for (int i = 0; i < 5; ++i) {
        check(xs[i]);
        if (i == 0 || i == 3) {
            TORCH_CHECK(xs[i].sizes() == x.sizes(), "activation shape mismatch");
        } else {
            TORCH_CHECK(xs[i].dim() == 1 && xs[i].numel() == C, "parameter shape mismatch");
        }
        dx.push_back(at::empty_like(xs[i]));
    }
    for (auto g : grads) {
        check(g);
        TORCH_CHECK(!g.numel() || g.sizes() == x.sizes(), "gradient shape mismatch");
    }
    if (!rows) {
        for (int i : {1, 2, 4})
            dx[i].zero_();
        return dx;
    }
    auto partial = at::empty({rows, 3, C}, x.options().dtype(at::kFloat));
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "key_gate_backward", [&] {
            KeyArgs<scalar_t> a{};
            for (int i = 0; i < 5; ++i) {
                a.x[i] = xs[i].data_ptr<scalar_t>();
                a.dx[i] = dx[i].data_ptr<scalar_t>();
            }
            for (int i = 0; i < 3; ++i)
                a.g[i] = grads[i].numel() ? grads[i].data_ptr<scalar_t>() : nullptr;
            key_backward_rows<scalar_t>
                <<<rows * H, 32, 0, stream>>>(a, partial.data_ptr<float>(), C, H);
            key_backward_parameters<scalar_t>
                <<<(C + 255) / 256, 256, 0, stream>>>(a, partial.data_ptr<float>(), rows, C);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dx;
}

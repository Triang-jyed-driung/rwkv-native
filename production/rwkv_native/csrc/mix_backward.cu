// Token-shift/mixing adjoints. Each input gradient has one writer. Coefficient
// gradients use fixed row chunks and an FP32 final reduction, without atomics.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
using at::Tensor;
template <typename T, typename P> struct MixArgs {
    const P *c[6];
    const T *g[6];
    P *dc[6];
    int count;
};
template <typename T, typename P>
__global__ void mix_backward_inputs(MixArgs<T, P> a, T *dx, T *ds, const T *final_grad, int B, int Tn,
                                    int C) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(B) * Tn * C)
        return;
    const int c = i % C, row = i / C, t = row % Tn, b = row / Tn;
    float d = 0.f, prev = 0.f;
    for (int g = 0; g < a.count; ++g) {
        if (!a.g[g])
            continue;
        const float coeff = float(a.c[g][c]), dy = float(a.g[g][i]);
        d += dy * (1.f - coeff);
        if (t + 1 < Tn)
            d += float(a.g[g][i + C]) * coeff;
        if (t == 0)
            prev += dy * coeff;
    }
    if (t + 1 == Tn && final_grad)
        d += float(final_grad[int64_t(b) * C + c]);
    dx[i] = T(d);
    if (t == 0)
        ds[int64_t(b) * C + c] = T(prev);
}
template <typename T, typename P>
__global__ void mix_backward_coefficients(MixArgs<T, P> a, const T *x, const T *state, float *partial,
                                          int rows, int Tn, int C, int chunks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, chunk = blockIdx.y, group = blockIdx.z;
    if (c >= C)
        return;
    float total = 0.f;
    if (a.g[group])
        for (int row = chunk * 128; row < min(rows, (chunk + 1) * 128); ++row) {
            const int t = row % Tn, b = row / Tn;
            const float prev =
                t ? float(x[int64_t(row - 1) * C + c]) : float(state[int64_t(b) * C + c]);
            total = fmaf(prev - float(x[int64_t(row) * C + c]),
                         float(a.g[group][int64_t(row) * C + c]), total);
        }
    partial[(int64_t(group) * chunks + chunk) * C + c] = total;
}
template <typename T, typename P>
__global__ void mix_backward_reduce(MixArgs<T, P> a, const float *partial, int C, int chunks) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, g = blockIdx.y;
    if (c >= C)
        return;
    float total = 0.f;
    for (int chunk = 0; chunk < chunks; ++chunk)
        total += partial[(int64_t(g) * chunks + chunk) * C + c];
    a.dc[g][c] = P(total);
}
std::vector<Tensor> mix_backward(Tensor x, Tensor state, std::vector<Tensor> coeff,
                                 std::vector<Tensor> grad, Tensor final_grad) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 3 && x.size(1) > 0 && x.size(2) > 0,
                "expected nonempty contiguous CUDA [B,T,C]");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "unsupported input dtype");
    TORCH_CHECK(coeff.size() >= 1 && coeff.size() <= 6 && grad.size() == coeff.size(),
                "expected 1..6 coefficient/gradient pairs");
    TORCH_CHECK(coeff[0].scalar_type() == at::kFloat || coeff[0].scalar_type() == at::kHalf ||
                    coeff[0].scalar_type() == at::kBFloat16, "unsupported coefficient dtype");
    c10::cuda::CUDAGuard guard(x.device());
    const int B = x.size(0), Tn = x.size(1), C = x.size(2), rows = B * Tn,
              chunks = (rows + 127) / 128;
    auto check = [&](const Tensor &t) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type() &&
                        t.is_contiguous(),
                    "dtype/device/layout mismatch");
    };
    check(state);
    check(final_grad);
    TORCH_CHECK(state.sizes() == at::IntArrayRef({B, C}), "invalid initial state");
    TORCH_CHECK(final_grad.numel() == 0 || final_grad.sizes() == state.sizes(),
                "invalid final-state gradient");
    std::vector<Tensor> result = {at::empty_like(x), at::empty_like(state)};
    for (size_t i = 0; i < coeff.size(); ++i) {
        TORCH_CHECK(coeff[i].device() == x.device() && coeff[i].is_contiguous() &&
                        coeff[i].scalar_type() == coeff[0].scalar_type(),
                    "coefficient dtype/device/layout mismatch");
        check(grad[i]);
        TORCH_CHECK(coeff[i].dim() == 1 && coeff[i].numel() == C, "invalid coefficient shape");
        TORCH_CHECK(grad[i].numel() == 0 || grad[i].sizes() == x.sizes(), "invalid mixed gradient");
        result.push_back(at::empty_like(coeff[i]));
    }
    if (!rows) {
        for (size_t i = 2; i < result.size(); ++i)
            result[i].zero_();
        return result;
    }
    auto partial = at::empty({int64_t(coeff.size()), chunks, C}, x.options().dtype(at::kFloat));
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "mix_backward", [&] {
        using X = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, coeff[0].scalar_type(),
                                       "mix_backward_parameters", [&] {
        using P = scalar_t;
        MixArgs<X, P> a{};
        a.count = coeff.size();
        for (size_t i = 0; i < coeff.size(); ++i) {
            a.c[i] = coeff[i].data_ptr<scalar_t>();
            a.g[i] = grad[i].numel() ? grad[i].data_ptr<X>() : nullptr;
            a.dc[i] = result[i + 2].data_ptr<scalar_t>();
        }
        mix_backward_inputs<X, P><<<(x.numel() + 255) / 256, 256, 0, stream>>>(
            a, result[0].data_ptr<X>(), result[1].data_ptr<X>(),
            final_grad.numel() ? final_grad.data_ptr<X>() : nullptr, B, Tn, C);
        mix_backward_coefficients<X, P>
            <<<dim3((C + 255) / 256, chunks, coeff.size()), 256, 0, stream>>>(
                a, x.data_ptr<X>(), state.data_ptr<X>(), partial.data_ptr<float>(),
                rows, Tn, C, chunks);
        mix_backward_reduce<X, P><<<dim3((C + 255) / 256, coeff.size()), 256, 0, stream>>>(
            a, partial.data_ptr<float>(), C, chunks);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

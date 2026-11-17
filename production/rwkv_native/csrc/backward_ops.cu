// FP32 arithmetic for activation and normalization adjoints. No half->float
// round trips are inserted into forward computations. Gradients are narrowed
// only once when stored in the corresponding input/parameter dtype.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
using at::Tensor;

__device__ float activation(float x, int kind) {
    if (kind == 1)
        return x > 0 ? x * x : 0.f;
    if (kind == 2)
        return tanhf(x);
    if (kind == 3)
        return 1.f / (1.f + expf(-x));
    return x;
}
__device__ float derivative(float x, int kind) {
    if (kind == 1)
        return x > 0 ? 2.f * x : 0.f;
    const float y = activation(x, kind);
    if (kind == 2)
        return 1.f - y * y;
    if (kind == 3)
        return y * (1.f - y);
    return 1.f;
}
template <typename T>
__global__ void act_forward_kernel(const T *x, float *y, int64_t n, int kind) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = activation(float(x[i]), kind);
}
template <typename T, typename G>
__global__ void act_backward_kernel(const T *x, const G *g, T *dx, int64_t n, int kind) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        dx[i] = T(float(g[i]) * derivative(float(x[i]), kind));
}
static void check_float_tensor(const Tensor &t) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "contiguous CUDA tensor required");
    TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf ||
                    t.scalar_type() == at::kBFloat16,
                "expected FP32/FP16/BF16");
}
Tensor activation_forward(Tensor x, int64_t kind) {
    check_float_tensor(x);
    TORCH_CHECK(kind >= 0 && kind <= 3, "invalid activation");
    c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty(x.sizes(), x.options().dtype(at::kFloat));
    if (x.numel()) {
        auto stream = c10::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, x.scalar_type(), "activation_forward", [&] {
                act_forward_kernel<scalar_t><<<(x.numel() + 255) / 256, 256, 0, stream>>>(
                    x.data_ptr<scalar_t>(), y.data_ptr<float>(), x.numel(), kind);
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}
Tensor activation_backward(Tensor x, Tensor grad, int64_t kind) {
    check_float_tensor(x);
    check_float_tensor(grad);
    TORCH_CHECK(x.sizes() == grad.sizes() && x.device() == grad.device(), "gradient mismatch");
    TORCH_CHECK(kind >= 0 && kind <= 3, "invalid activation");
    c10::cuda::CUDAGuard guard(x.device());
    auto dx = at::empty_like(x);
    if (x.numel()) {
        auto stream = c10::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, x.scalar_type(), "activation_backward_x", [&] {
                using X = scalar_t;
                AT_DISPATCH_FLOATING_TYPES_AND2(
                    at::kHalf, at::kBFloat16, grad.scalar_type(), "activation_backward_g", [&] {
                        act_backward_kernel<X, scalar_t>
                            <<<(x.numel() + 255) / 256, 256, 0, stream>>>(
                                x.data_ptr<X>(), grad.data_ptr<scalar_t>(), dx.data_ptr<X>(),
                                x.numel(), kind);
                    });
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return dx;
}

__device__ float row_sum(float value, float *scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int d = 16; d; d >>= 1)
        value += __shfl_down_sync(0xffffffff, value, d);
    if (lane == 0)
        scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? scratch[threadIdx.x] : 0.f;
    if (warp == 0) {
        for (int d = 16; d; d >>= 1)
            value += __shfl_down_sync(0xffffffff, value, d);
        if (lane == 0)
            scratch[8] = value;
    }
    __syncthreads();
    float result = scratch[8];
    __syncthreads();
    return result;
}
template <typename X, typename W>
__global__ void ln_backward_rows(const X *x, const W *w, const X *g, X *dx, float *partial, int C,
                                 float eps, const X *residual = nullptr, const X *direct = nullptr,
                                 int tokens = 0, const int64_t *selected = nullptr) {
    const int row = blockIdx.x;
    if (tokens && selected)
        CUDA_KERNEL_ASSERT(selected[row] >= 0 && selected[row] < tokens);
    const int64_t input_row =
        tokens ? int64_t(row) * tokens + (selected ? selected[row] : tokens - 1) : row;
    const int64_t base = input_row * C, gradbase = int64_t(row) * C;
    __shared__ float scratch[9];
    float sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256)
        sum += float(x[base + c]) + (residual ? float(residual[base + c]) : 0.f);
    const float mean = row_sum(sum, scratch) / C;
    sum = 0.f;
    for (int c = threadIdx.x; c < C; c += 256) {
        float d = float(x[base + c]) + (residual ? float(residual[base + c]) : 0.f) - mean;
        sum = fmaf(d, d, sum);
    }
    const float inv = rsqrtf(row_sum(sum, scratch) / C + eps);
    float a = 0.f, b = 0.f;
    for (int c = threadIdx.x; c < C; c += 256) {
        const float xhat =
                        (float(x[base + c]) + (residual ? float(residual[base + c]) : 0.f) - mean) *
                        inv,
                    dy = float(g[gradbase + c]);
        const float u = dy * float(w[c]);
        a += u;
        b = fmaf(u, xhat, b);
        partial[(int64_t(row) * 2) * C + c] = dy * xhat;
        partial[(int64_t(row) * 2 + 1) * C + c] = dy;
    }
    a = row_sum(a, scratch) / C;
    b = row_sum(b, scratch) / C;
    for (int c = threadIdx.x; c < C; c += 256) {
        const float xhat =
            (float(x[base + c]) + (residual ? float(residual[base + c]) : 0.f) - mean) * inv;
        dx[base + c] = X(inv * (float(g[gradbase + c]) * float(w[c]) - a - xhat * b) +
                         (direct ? float(direct[base + c]) : 0.f));
    }
}

template <typename W>
__global__ void ln_backward_parameters(const float *partial, W *dw, W *db, int rows, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float w = 0.f, b = 0.f;
    for (int row = 0; row < rows; ++row) {
        w += partial[(int64_t(row) * 2) * C + c];
        b += partial[(int64_t(row) * 2 + 1) * C + c];
    }
    dw[c] = W(w);
    db[c] = W(b);
}
std::vector<Tensor> layer_norm_backward(Tensor x, Tensor weight, Tensor grad, double eps) {
    check_float_tensor(x);
    check_float_tensor(weight);
    check_float_tensor(grad);
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0 && weight.dim() == 1 && weight.numel() == x.size(-1),
                "invalid LayerNorm shapes");
    TORCH_CHECK(grad.sizes() == x.sizes() && grad.scalar_type() == x.scalar_type(),
                "gradient must match activation dtype");
    TORCH_CHECK(x.device() == weight.device() && x.device() == grad.device(), "device mismatch");
    TORCH_CHECK(eps > 0, "positive epsilon required");
    c10::cuda::CUDAGuard guard(x.device());
    const int C = x.size(-1), rows = x.numel() / C;
    auto dx = at::empty_like(x), dw = at::zeros_like(weight), db = at::zeros_like(weight);
    if (!rows)
        return {dx, dw, db};
    auto partial = at::empty({rows, 2, C}, x.options().dtype(at::kFloat));
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "ln_backward_x", [&] {
            using X = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, weight.scalar_type(), "ln_backward_w", [&] {
                    ln_backward_rows<X, scalar_t><<<rows, 256, 0, stream>>>(
                        x.data_ptr<X>(), weight.data_ptr<scalar_t>(), grad.data_ptr<X>(),
                        dx.data_ptr<X>(), partial.data_ptr<float>(), C, eps);
                    ln_backward_parameters<scalar_t><<<(C + 255) / 256, 256, 0, stream>>>(
                        partial.data_ptr<float>(), dw.data_ptr<scalar_t>(), db.data_ptr<scalar_t>(),
                        rows, C);
                });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dw, db};
}

template <typename T, typename P, typename W>
__global__ void vres_backward_kernel(const T *v, const T *first, const W *bias, const P *logits,
                                     const T *g, T *dv, T *df, P *dp, float *partial, int64_t n,
                                     int C) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float gate = 1.f / (1.f + expf(-(float(bias[i % C]) + float(logits[i]))));
    const float dy = float(g[i]);
    const float d = dy * (float(first[i]) - float(v[i])) * gate * (1.f - gate);
    dv[i] = T(dy * (1.f - gate));
    df[i] = T(dy * gate);
    dp[i] = P(d);
    partial[i] = d;
}
template <typename T>
__global__ void sum_columns_kernel(const float *x, T *y, int64_t rows, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float total = 0.f;
    for (int64_t row = 0; row < rows; ++row)
        total += x[row * C + c];
    y[c] = T(total);
}
std::vector<Tensor> vres_backward(Tensor v, Tensor first, Tensor bias, Tensor logits, Tensor grad) {
    for (auto t : {v, first, bias, logits, grad}) {
        check_float_tensor(t);
        TORCH_CHECK(t.device() == v.device(), "device mismatch");
    }
    TORCH_CHECK(v.dim() >= 2 && v.size(-1) > 0 && first.sizes() == v.sizes() &&
                    logits.sizes() == v.sizes() && grad.sizes() == v.sizes(),
                "shape mismatch");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == v.size(-1), "invalid bias");
    TORCH_CHECK(first.scalar_type() == v.scalar_type() &&
                    grad.scalar_type() == v.scalar_type(),
                "dtype mismatch");
    c10::cuda::CUDAGuard guard(v.device());
    auto dv = at::empty_like(v), df = at::empty_like(first), dp = at::empty_like(logits),
         db = at::zeros_like(bias);
    if (!v.numel())
        return {dv, df, db, dp};
    auto partial = at::empty(v.sizes(), v.options().dtype(at::kFloat));
    auto stream = c10::cuda::getCurrentCUDAStream();
    const int C = v.size(-1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, v.scalar_type(), "vres_backward_v", [&] {
            using V = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, bias.scalar_type(), "vres_backward_w", [&] {
                    using W = scalar_t;
                    AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::kHalf, at::kBFloat16, logits.scalar_type(), "vres_backward_p", [&] {
                            vres_backward_kernel<V, scalar_t, W>
                                <<<(v.numel() + 255) / 256, 256, 0, stream>>>(
                                    v.data_ptr<V>(), first.data_ptr<V>(), bias.data_ptr<W>(),
                                    logits.data_ptr<scalar_t>(), grad.data_ptr<V>(), dv.data_ptr<V>(),
                                    df.data_ptr<V>(), dp.data_ptr<scalar_t>(), partial.data_ptr<float>(),
                                    v.numel(), C);
                        });
                    sum_columns_kernel<W><<<(C + 255) / 256, 256, 0, stream>>>(
                        partial.data_ptr<float>(), db.data_ptr<W>(), v.numel() / C, C);
                });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dv, df, db, dp};
}

__device__ float warp_total(float v) {
    for (int d = 16; d; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    return __shfl_sync(0xffffffff, v, 0);
}
template <typename T, typename P> struct ReadoutArgs {
    const T *x[8];
    T *dx[8];
    const P *p[8];
    P *dp[8];
};
template <typename T, typename P>
__global__ __launch_bounds__(32, 2) void readout_backward_rows(ReadoutArgs<T, P> a, const T *grad,
                                                               float *partial, int C, int H) {
    const int group = blockIdx.x, row = group / H, head = group % H, lane = threadIdx.x;
    float x[2], r[2], k[2], v[2], scale[2], w[2], bias[2], gate[2], dy[2], hat[2], dn[2];
    float sum = 0.f, dot = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int64_t i = int64_t(group) * 64 + lane + j * 32;
        const int c = head * 64 + lane + j * 32;
        x[j] = float(a.x[0][i]);
        r[j] = float(a.x[1][i]);
        k[j] = float(a.x[2][i]);
        v[j] = float(a.x[3][i]);
        scale[j] = float(a.p[4][c]);
        w[j] = float(a.p[5][c]);
        bias[j] = float(a.p[6][c]);
        gate[j] = float(a.x[7][i]);
        dy[j] = float(grad[i]);
        sum += x[j];
        dot += r[j] * k[j] * scale[j];
    }
    const float mean = warp_total(sum) / 64.f;
    sum = 0.f;
    for (int j = 0; j < 2; ++j) {
        float d = x[j] - mean;
        sum += d * d;
    }
    const float inv = rsqrtf(warp_total(sum) / 64.f + 64e-5f);
    dot = warp_total(dot);
    float ddot = 0.f, du = 0.f, dux = 0.f;
    for (int j = 0; j < 2; ++j) {
        hat[j] = (x[j] - mean) * inv;
        dn[j] = dy[j] * gate[j];
        ddot += dn[j] * v[j];
        du += dn[j] * w[j];
        dux += dn[j] * w[j] * hat[j];
    }
    ddot = warp_total(ddot);
    du = warp_total(du) / 64.f;
    dux = warp_total(dux) / 64.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int64_t i = int64_t(group) * 64 + lane + j * 32;
        const int c = head * 64 + lane + j * 32;
        a.dx[0][i] = T(inv * (dn[j] * w[j] - du - hat[j] * dux));
        a.dx[1][i] = T(ddot * k[j] * scale[j]);
        a.dx[2][i] = T(ddot * r[j] * scale[j]);
        a.dx[3][i] = T(dn[j] * dot);
        a.dx[7][i] = T(dy[j] * (hat[j] * w[j] + bias[j] + dot * v[j]));
        partial[(int64_t(row) * 3) * C + c] = ddot * r[j] * k[j];
        partial[(int64_t(row) * 3 + 1) * C + c] = dn[j] * hat[j];
        partial[(int64_t(row) * 3 + 2) * C + c] = dn[j];
    }
}
template <typename T, typename P>
__global__ void readout_backward_parameters(const float *p, ReadoutArgs<T, P> a, int rows, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float ds = 0.f, dw = 0.f, db = 0.f;
    for (int row = 0; row < rows; ++row) {
        ds += p[(int64_t(row) * 3) * C + c];
        dw += p[(int64_t(row) * 3 + 1) * C + c];
        db += p[(int64_t(row) * 3 + 2) * C + c];
    }
    a.dp[4][c] = P(ds);
    a.dp[5][c] = P(dw);
    a.dp[6][c] = P(db);
}
std::vector<Tensor> readout_backward(std::vector<Tensor> xs, Tensor grad, int64_t H) {
    TORCH_CHECK(xs.size() == 8, "expected x,r,k,v,scale,weight,bias,gate");
    auto x = xs[0];
    check_float_tensor(x);
    check_float_tensor(grad);
    TORCH_CHECK(x.dim() >= 2 && H > 0 && x.size(-1) == H * 64, "expected 64-wide heads");
    TORCH_CHECK(grad.device() == x.device() && grad.scalar_type() == x.scalar_type() &&
                    grad.sizes() == x.sizes(),
                "invalid output gradient");
    TORCH_CHECK(xs[4].scalar_type() == at::kFloat || xs[4].scalar_type() == at::kHalf ||
                    xs[4].scalar_type() == at::kBFloat16, "unsupported parameter dtype");
    c10::cuda::CUDAGuard guard(x.device());
    std::vector<Tensor> dx;
    const int C = x.size(-1), rows = x.numel() / C;
    for (int i = 0; i < 8; ++i) {
        check_float_tensor(xs[i]);
        TORCH_CHECK(xs[i].device() == x.device() && xs[i].scalar_type() == ((i >= 4 && i <= 6) ? xs[4].scalar_type() : x.scalar_type()),
                    "input dtype/device mismatch");
        if (i >= 4 && i <= 6) {
            TORCH_CHECK(xs[i].dim() == 1 && xs[i].numel() == C, "invalid parameter shape");
        } else {
            TORCH_CHECK(xs[i].sizes() == x.sizes(), "invalid activation shape");
        }
        dx.push_back(at::empty_like(xs[i]));
    }
    if (!rows) {
        for (int i = 4; i <= 6; ++i)
            dx[i].zero_();
        return dx;
    }
    auto partial = at::empty({rows, 3, C}, x.options().dtype(at::kFloat));
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "readout_backward", [&] {
            using X = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xs[4].scalar_type(),
                                           "readout_backward_parameters", [&] {
            using P = scalar_t;
            ReadoutArgs<X, P> a{};
            for (int i = 0; i < 8; ++i) {
                if (i == 4 || i == 5 || i == 6) {
                    a.p[i] = xs[i].data_ptr<P>();
                    a.dp[i] = dx[i].data_ptr<P>();
                } else {
                    a.x[i] = xs[i].data_ptr<X>();
                    a.dx[i] = dx[i].data_ptr<X>();
                }
            }
            readout_backward_rows<X, P><<<rows * H, 32, 0, stream>>>(
                a, grad.data_ptr<X>(), partial.data_ptr<float>(), C, H);
            readout_backward_parameters<X, P>
                <<<(C + 255) / 256, 256, 0, stream>>>(partial.data_ptr<float>(), a, rows, C);
            });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dx;
}

// FP16 actual GEMM inputs, FP32 outputs and deterministic K8/K16 partition sums.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include "decode_ptx.cuh"
using at::Tensor;
struct PreGroups {
    const half_t *x[7];
    const half_t *w[7];
    int n[7], splits[7], prefix[7];
    int count, total_n;
};
#include "pre_partial_m16.cuh"
#include "pre_partial_m64.cuh"
#include "pre_fused_m64.cuh"

__global__ void pre_reduce(PreGroups p, const float *partial, float *out, int64_t rows) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= int64_t(p.total_n) * rows)
        return;
    int g = 0;
    while (g + 1 < p.count && i >= int64_t(p.prefix[g + 1]) * rows)
        ++g;
    float sum = 0;
    for (int part = 0; part < p.splits[g]; ++part)
        sum = __fadd_rn(sum, partial[int64_t(part) * p.total_n * rows + i]);
    out[i] = sum;
}

Tensor pre_group(std::vector<Tensor> xs, std::vector<Tensor> weights, int64_t mode) {
    TORCH_CHECK(xs.size() == weights.size() && (xs.size() == 1 || xs.size() == 6 || xs.size() == 7),
                "expected one projection or R/K/V/W/A/G/(V) groups");
    auto x = xs[0];
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kHalf,
                "expected CUDA FP16 matrices");
    c10::cuda::CUDAGuard guard(x.device());
    int64_t M = x.size(0);
    PreGroups p{};
    p.count = xs.size();
    int tasks = 0, tiles = 0;
    const int expected[] = {2048, 2048, 2048, 96, 96, 256, 64};
    for (int g = 0; g < p.count; ++g) {
        const auto a = xs[g], w = weights[g];
        TORCH_CHECK(a.device() == x.device() && a.scalar_type() == at::kHalf && a.is_contiguous() &&
                        a.sizes() == at::IntArrayRef({M, 2048}),
                    "expected contiguous FP16 [M,2048] inputs");
        TORCH_CHECK(w.device() == x.device() && w.scalar_type() == at::kHalf && w.is_contiguous() &&
                        w.dim() == 2 && w.size(1) == 2048,
                    "expected contiguous FP16 [N,2048] weights");
        const int N = w.size(0);
        TORCH_CHECK((p.count == 1 && (N == 64 || N == 96 || N == 256 || N == 2048)) ||
                        (p.count != 1 && N == expected[g]),
                    "invalid output width");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                        reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                    "unaligned operands");
        p.x[g] = (const half_t *)a.data_ptr();
        p.w[g] = (const half_t *)w.data_ptr();
        p.n[g] = N;
        p.splits[g] = (N == 96 || N == 256) ? 16 : 8;
        p.prefix[g] = p.total_n;
        p.total_n += N;
        tiles += (N + 63) / 64;
        tasks += ((N + 63) / 64) * p.splits[g];
    }
    TORCH_CHECK(mode == -1 || mode == 0 || mode == 16 || mode == 64, "invalid launch mode");
    if (mode == -1)
        mode = M <= 16 ? 16 : M <= 96 ? 64 : 0;
    auto out = at::empty({int64_t(p.total_n) * M}, x.options().dtype(at::kFloat));
    if (M) {
        int tile = mode == 16 ? 16 : 64;
        TORCH_CHECK((M + tile - 1) / tile <= 65535, "too many rows");
        auto stream = c10::cuda::getCurrentCUDAStream();
        if (!mode)
            pre_fused_m64<<<dim3(tiles, (M + 63) / 64), 128, 16384, stream>>>(
                p, out.data_ptr<float>(), M);
        else {
            auto partial = at::empty({16, int64_t(p.total_n) * M}, out.options());
            if (mode == 16)
                pre_partial_m16<<<dim3(tasks, (M + 15) / 16), 128, 10240, stream>>>(
                    p, partial.data_ptr<float>(), M);
            else
                pre_partial_m64<<<dim3(tasks, (M + 63) / 64), 128, 16384, stream>>>(
                    p, partial.data_ptr<float>(), M);
            pre_reduce<<<(int64_t(p.total_n) * M + 255) / 256, 256, 0, stream>>>(
                p, partial.data_ptr<float>(), out.data_ptr<float>(), M);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

// Grouped low-rank expansion adapted from FlashRWKV2, with FP32 activation edges.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include "rank_group_launch.cuh"
using at::Tensor;

struct PackGroups {
    const float *x[4];
    at::Half *out[4];
    int64_t sizes[4];
    int count;
};
__global__ __launch_bounds__(256, 2) void pack_rank_kernel(PackGroups p, int64_t total) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= total)
        return;
    int group = 0;
    while (group + 1 < p.count && i >= p.sizes[group]) {
        i -= p.sizes[group];
        ++group;
    }
    float value = p.x[group][i];
    if (group == 0)
        value = tanhf(value);
    else if (group == 2)
        value = 1.f / (1.f + expf(-value));
    p.out[group][i] = at::Half(value);
}
std::vector<Tensor> pack_rank_inputs(std::vector<Tensor> xs) {
    TORCH_CHECK(xs.size() == 3 || xs.size() == 4, "expected W/A/G/(V) branches");
    const auto x = xs[0];
    TORCH_CHECK(x.is_cuda(), "expected CUDA inputs");
    c10::cuda::CUDAGuard guard(x.device());
    PackGroups p{};
    p.count = xs.size();
    int64_t total = 0;
    std::vector<Tensor> ys;
    for (size_t g = 0; g < xs.size(); ++g) {
        TORCH_CHECK(xs[g].device() == x.device() && xs[g].scalar_type() == at::kFloat &&
                        xs[g].is_contiguous(),
                    "expected contiguous FP32 branches on the same device");
        ys.push_back(at::empty_like(xs[g], xs[g].options().dtype(at::kHalf)));
        p.x[g] = xs[g].data_ptr<float>();
        p.out[g] = ys.back().data_ptr<at::Half>();
        p.sizes[g] = xs[g].numel();
        total += p.sizes[g];
    }
    if (total) {
        pack_rank_kernel<<<(total + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(p,
                                                                                             total);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return ys;
}

Tensor grouped_rank_projection(std::vector<Tensor> xs, std::vector<Tensor> weights, int64_t tile) {
    TORCH_CHECK(xs.size() == weights.size() && (xs.size() == 3 || xs.size() == 4),
                "expected W/A/G/(V) operands and weights");
    TORCH_CHECK(tile == 16 || tile == 64, "invalid row tile");
    auto x = xs[0];
    TORCH_CHECK(x.is_cuda() && x.dim() == 2, "expected CUDA matrices");
    c10::cuda::CUDAGuard guard(x.device());
    const int64_t M = x.size(0), N = 2048, count = xs.size();
    const int ranks[] = {96, 96, 256, 64};
    for (int g = 0; g < count; ++g) {
        TORCH_CHECK(xs[g].device() == x.device() && xs[g].scalar_type() == at::kHalf &&
                        xs[g].is_contiguous() && xs[g].sizes() == at::IntArrayRef({M, ranks[g]}),
                    "invalid packed rank input");
        TORCH_CHECK(weights[g].device() == x.device() && weights[g].scalar_type() == at::kHalf &&
                        weights[g].is_contiguous() &&
                        weights[g].sizes() == at::IntArrayRef({N, ranks[g]}),
                    "expected contiguous [2048,rank] weights");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(xs[g].data_ptr()) % 16 == 0 &&
                        reinterpret_cast<uintptr_t>(weights[g].data_ptr()) % 16 == 0,
                    "unaligned rank operand");
    }
    auto flat = at::empty({count * M * N}, x.options().dtype(at::kFloat));
    if (M) {
        TORCH_CHECK((M + tile - 1) / tile <= 65535, "too many rows");
        launch_rank_group(xs, weights, flat, tile, c10::cuda::getCurrentCUDAStream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return flat;
}

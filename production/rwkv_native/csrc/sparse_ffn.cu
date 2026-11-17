// SPDX-License-Identifier: Apache-2.0
// Adapted from BlinkDL/Albatross faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// commit 545dd1870ea6cf424d7f7d4da16b7f87491405db, cmix_sparse_spmv_relu_*.
// Modifications: FP32 preactivation, FP16/BF16 packed operands, FP32 FMA,
// explicit partials with ordered reduction (no low-precision atomics), tails.
// See ../third_party/ALBATROSS_LICENSE.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

using at::Tensor;
constexpr int THREADS = 128;
constexpr int FEATURES = 128;

template<typename P> __device__ __forceinline__ float2 pair(const P *p);
template<> __device__ __forceinline__ float2 pair(const at::Half *p) {
    return __half22float2(*reinterpret_cast<const __half2 *>(p));
}
template<> __device__ __forceinline__ float2 pair(const at::BFloat16 *p) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(p));
}

template<typename P>
__global__ __launch_bounds__(THREADS, 4) void sparse_down(
        const float *__restrict__ preactivation, const P *__restrict__ weight,
        float *__restrict__ partial, int F, int C, int tiles) {
    __shared__ float values[FEATURES];
    __shared__ int indices[FEATURES], counts[4], offsets[4], count;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int feature_base = blockIdx.x * FEATURES, row = blockIdx.z;
    int feature = feature_base + tid;
    float positive = feature < F ? fmaxf(preactivation[int64_t(row) * F + feature], 0.f) : 0.f;
    // Exactly the existing relu_pack boundary: square in FP32, then narrow.
    float value = float(P(positive * positive));
    values[tid] = value;
    bool nonzero = value != 0.f;
    unsigned ballot = __ballot_sync(0xffffffff, nonzero);
    int position = __popc(ballot & ((1u << lane) - 1u));
    if (!lane)
        counts[warp] = __popc(ballot);
    __syncthreads();
    if (!tid) {
        int total = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            offsets[w] = total;
            total += counts[w];
        }
        count = total;
    }
    __syncthreads();
    if (nonzero)
        indices[offsets[warp] + position] = tid;
    __syncthreads();

    int column = blockIdx.y * (2 * THREADS) + 2 * tid;
    if (column >= C)
        return;
    float2 acc0 = {0.f, 0.f}, acc1 = {0.f, 0.f};
    for (int i = 0; i < count; i += 2) {
        int local = indices[i];
        float2 matrix = pair(weight + int64_t(feature_base + local) * C + column);
        acc0.x = fmaf(values[local], matrix.x, acc0.x);
        acc0.y = fmaf(values[local], matrix.y, acc0.y);
        if (i + 1 < count) {
            local = indices[i + 1];
            matrix = pair(weight + int64_t(feature_base + local) * C + column);
            acc1.x = fmaf(values[local], matrix.x, acc1.x);
            acc1.y = fmaf(values[local], matrix.y, acc1.y);
        }
    }
    int64_t offset = (int64_t(row) * tiles + blockIdx.x) * C + column;
    *reinterpret_cast<float2 *>(partial + offset) = {acc0.x + acc1.x, acc0.y + acc1.y};
}

template<bool WIDE>
__global__ void finish_sparse(const float *partial, float *out, int rows, int C, int tiles) {
    int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t(rows) * C)
        return;
    int row = index / C, column = index % C;
    int64_t base = int64_t(row) * tiles * C + column;
    using accumulator_t = std::conditional_t<WIDE,double,float>;
    accumulator_t total = partial[base];
    for (int tile = 1; tile < tiles; ++tile)
        total += partial[base + int64_t(tile) * C];
    out[index] = float(total);
}

Tensor sparse_ffn(Tensor x, Tensor weight) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kFloat && x.is_contiguous(),
                "sparse FFN expects contiguous FP32 [rows,F]");
    TORCH_CHECK(weight.device() == x.device() && weight.dim() == 2 && weight.is_contiguous() &&
                    (weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kBFloat16),
                "sparse FFN expects contiguous FP16/BF16 [F,C] weights");
    int rows = x.size(0), F = x.size(1), C = weight.size(1), tiles = (F + FEATURES - 1) / FEATURES;
    TORCH_CHECK(rows > 0 && rows <= 4 && F > 0 && weight.size(0) == F && C > 0 && C % 2 == 0 &&
                    reinterpret_cast<uintptr_t>(weight.data_ptr()) % 4 == 0,
                "sparse FFN requires 1..4 rows, matching F, even C and 4-byte aligned weights");
    c10::cuda::CUDAGuard guard(x.device());
    auto partial = at::empty({rows, tiles, C}, x.options());
    auto out = at::empty({rows, C}, x.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    dim3 grid(tiles, (C + 2 * THREADS - 1) / (2 * THREADS), rows);
    if (weight.scalar_type() == at::kHalf)
        sparse_down<at::Half><<<grid, THREADS, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<at::Half>(), partial.data_ptr<float>(), F, C, tiles);
    else
        sparse_down<at::BFloat16><<<grid, THREADS, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<at::BFloat16>(), partial.data_ptr<float>(), F, C, tiles);
    if (weight.scalar_type() == at::kHalf)
        finish_sparse<true><<<(int64_t(rows) * C + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), out.data_ptr<float>(), rows, C, tiles);
    else
        finish_sparse<false><<<(int64_t(rows) * C + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(), out.data_ptr<float>(), rows, C, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &sparse_ffn); }

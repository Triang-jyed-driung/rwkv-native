// Fixed-order FP16/BF16 GEMM with FP32 accumulators/output.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"
using at::Tensor;

__global__ void reduce_splits(const float4 *partial, float4 *output, int64_t n, int splits) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float4 sum = partial[i];
    for (int s = 1; s < splits; ++s) {
        const float4 p = partial[int64_t(s) * n + i];
        sum.x += p.x;
        sum.y += p.y;
        sum.z += p.z;
        sum.w += p.w;
    }
    output[i] = sum;
}

template <typename P, int MTile>
void run_fixed(Tensor x, Tensor w, Tensor partial, int M, int N, int K, int splits) {
    using Row = cutlass::layout::RowMajor;
    using Col = cutlass::layout::ColumnMajor;
    using Epilogue =
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float,
                                                     cutlass::epilogue::thread::ScaleType::Nothing>;
    using Gemm = cutlass::gemm::device::GemmBatched<
        P, Row, P, Col, float, Row, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<MTile, 128, 32>, cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3, 8, 8>;
    typename Gemm::Arguments args(
        {M, N, K / splits}, {reinterpret_cast<const P *>(x.data_ptr()), Row(K)}, K / splits,
        {reinterpret_cast<const P *>(w.data_ptr()), Col(K)}, K / splits, {nullptr, Row(N)}, 0,
        {partial.data_ptr<float>(), Row(N)}, int64_t(M) * N, {}, splits);
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
                "invalid fixed GEMM arguments");
    TORCH_CHECK(gemm(args, nullptr, c10::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess,
                "fixed GEMM launch failed");
}

Tensor fixed_projection(Tensor x, Tensor w, int64_t splits, int64_t tile) {
    TORCH_CHECK(x.is_cuda() && w.device() == x.device() && x.dim() == 2 && w.dim() == 2 &&
                    x.is_contiguous() && w.is_contiguous(),
                "expected contiguous CUDA [M,K], [N,K]");
    TORCH_CHECK(x.scalar_type() == w.scalar_type() &&
                    (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16),
                "invalid dtype");
    c10::cuda::CUDAGuard guard(x.device());
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K && N > 0 && N % 4 == 0 && splits > 0 && K > 0 && K % splits == 0 &&
                    (K / splits) % 32 == 0,
                "invalid shape/splits");
    TORCH_CHECK(tile == 32 || tile == 64 || tile == 128, "invalid M tile");
    auto out = at::empty({M, N}, x.options().dtype(at::kFloat));
    if (!M)
        return out;
    auto partial = splits == 1 ? out : at::empty({splits, M, N}, out.options());
    if (x.scalar_type() == at::kHalf) {
        if (tile == 32)
            run_fixed<cutlass::half_t, 32>(x, w, partial, M, N, K, splits);
        else if (tile == 64)
            run_fixed<cutlass::half_t, 64>(x, w, partial, M, N, K, splits);
        else
            run_fixed<cutlass::half_t, 128>(x, w, partial, M, N, K, splits);
    } else {
        if (tile == 32)
            run_fixed<cutlass::bfloat16_t, 32>(x, w, partial, M, N, K, splits);
        else if (tile == 64)
            run_fixed<cutlass::bfloat16_t, 64>(x, w, partial, M, N, K, splits);
        else
            run_fixed<cutlass::bfloat16_t, 128>(x, w, partial, M, N, K, splits);
    }
    if (splits > 1) {
        const int64_t n = int64_t(M) * N / 4;
        reduce_splits<<<(n + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4 *>(partial.data_ptr<float>()),
            reinterpret_cast<float4 *>(out.data_ptr<float>()), n, splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

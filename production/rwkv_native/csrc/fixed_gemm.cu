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

template <typename P, int MTile, int KTile = (MTile == 16 ? 64 : 32), int Stages = 3>
void run_fixed(Tensor x, Tensor w, Tensor partial, int M, int N, int K, int splits) {
    using Row = cutlass::layout::RowMajor;
    using Col = cutlass::layout::ColumnMajor;
    using Epilogue =
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float,
                                                     cutlass::epilogue::thread::ScaleType::Nothing>;
    using Gemm = cutlass::gemm::device::GemmBatched<
        P, Row, P, Col, float, Row, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<MTile, 128, KTile>,
        cutlass::gemm::GemmShape<MTile == 16 ? 16 : 32, 32, KTile>,
        cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages, 8, 8>;
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

template <typename P, int MTile>
void launch_fixed(Tensor x, Tensor w, Tensor partial, int M, int N, int K, int splits) {
    // Change staging only: keep the same ordered MMA K steps, split boundaries
    // and final FP32 reduction. Leave small batches on their existing schedule.
    if constexpr (MTile == 128) {
        if (M >= 128 && (K / splits) % 64 == 0) {
            run_fixed<P, MTile, 64, 3>(x, w, partial, M, N, K, splits);
            return;
        }
    } else if constexpr (MTile == 32 || MTile == 64) {
        if (M >= 128 && K >= 1024) {
            run_fixed<P, MTile, 32, 4>(x, w, partial, M, N, K, splits);
            return;
        }
    }
    run_fixed<P, MTile>(x, w, partial, M, N, K, splits);
}

template<class Kernel>
__global__ __launch_bounds__(Kernel::kThreadCount) void fused_splits(typename Kernel::Params p,int splits) {
    using Mma=typename Kernel::Mma;
    using Epilogue=typename Kernel::Epilogue;
    extern __shared__ char storage[];
    auto &shared=*reinterpret_cast<typename Kernel::SharedStorage*>(storage);
    int thread=threadIdx.x,warp=threadIdx.x/32,lane=threadIdx.x%32;
    cutlass::MatrixCoord offset_A(int(blockIdx.x)*Mma::Shape::kM,0);
    cutlass::MatrixCoord offset_B(0,int(blockIdx.y)*Mma::Shape::kN);
    typename Mma::FragmentC sum;
    sum.clear();
    for(int split=0;split<splits;++split) {
        typename Mma::IteratorA a(p.params_A,p.ref_A.data(),p.problem_size.mk(),thread,offset_A);
        typename Mma::IteratorB b(p.params_B,p.ref_B.data(),p.problem_size.kn(),thread,offset_B);
        a.add_pointer_offset(p.stride_A*split);
        b.add_pointer_offset(p.stride_B*split);
        Mma mma(shared.main_loop,thread,warp,lane);
        typename Mma::FragmentC accum;
        accum.clear();
        mma(p.gemm_k_iterations,accum,a,b,accum);
        if(split==0)sum=accum;
        else {
            #pragma unroll
            for(int i=0;i<Mma::FragmentC::kElements;++i)sum[i]+=accum[i];
        }
        __syncthreads();
    }
    cutlass::MatrixCoord offset(int(blockIdx.x)*Mma::Shape::kM,int(blockIdx.y)*Mma::Shape::kN);
    typename Epilogue::OutputTileIterator c(p.params_C,p.ref_C.data(),p.problem_size.mn(),thread,offset);
    typename Epilogue::OutputTileIterator d(p.params_D,p.ref_D.data(),p.problem_size.mn(),thread,offset);
    typename Kernel::OutputOp op(p.epilogue);
    Epilogue epilogue(shared.epilogue,thread,warp,lane);
    epilogue(op,d,sum,c);
}
template<class P,int MT>
void launch_fused_splits(Tensor x,Tensor w,Tensor out,int splits) {
    using Row=cutlass::layout::RowMajor;using Col=cutlass::layout::ColumnMajor;
    using Op=cutlass::epilogue::thread::LinearCombination<float,4,float,float,cutlass::epilogue::thread::ScaleType::Nothing>;
    using Gemm=cutlass::gemm::device::GemmBatched<P,Row,P,Col,float,Row,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<MT,128,64>,cutlass::gemm::GemmShape<32,32,64>,cutlass::gemm::GemmShape<16,8,16>,Op,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,3,8,8>;
    using Kernel=typename Gemm::GemmKernel;
    int M=x.size(0),K=x.size(1),N=w.size(0);
    cutlass::gemm::GemmCoord grid((M+MT-1)/MT,(N+127)/128,1);
    typename Kernel::Params p({M,N,K/splits},grid,{reinterpret_cast<P*>(x.data_ptr()),Row(K)},K/splits,
        {reinterpret_cast<P*>(w.data_ptr()),Col(K)},K/splits,{nullptr,Row(N)},0,{out.data_ptr<float>(),Row(N)},0,{},1);
    int smem=sizeof(typename Kernel::SharedStorage);
    if(smem>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(fused_splits<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    fused_splits<Kernel><<<dim3(grid.m(),grid.n()),Kernel::kThreadCount,smem,c10::cuda::getCurrentCUDAStream()>>>(p,splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor fixed_projection_impl(Tensor x, Tensor w, int64_t splits, int64_t tile, Tensor out) {
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
    TORCH_CHECK(tile == 16 || tile == 32 || tile == 64 || tile == 128, "invalid M tile");
    TORCH_CHECK(tile != 16 || (K / splits) % 64 == 0, "M16 requires K64-aligned splits");
    if (!out.defined())
        out = at::empty({M, N}, x.options().dtype(at::kFloat));
    TORCH_CHECK(out.device() == x.device() && out.scalar_type() == at::kFloat &&
                    out.sizes() == at::IntArrayRef({M, N}) && out.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "expected aligned contiguous FP32 output [M,N]");
    if (!M)
        return out;
    // Keep each original K segment and its FP32 sum order, but retain the
    // partial accumulators in registers rather than writing a split-K tensor.
    const bool fused = splits == 8 && (K / splits) % 64 == 0 &&
        ((M >= 768 && K == 10240 && N == 2560) ||
         ((M == 192 || M >= 256) && ((K == 4096 && (N == 4096 || N == 16384)) ||
                       (K == 16384 && N == 4096))));
    if (fused) {
        const bool large_tile = M >= 768 || (M != 192 && K == 4096 && N == 16384);
        if (x.scalar_type() == at::kHalf) {
            if (large_tile) launch_fused_splits<cutlass::half_t,128>(x,w,out,splits);
            else launch_fused_splits<cutlass::half_t,64>(x,w,out,splits);
        } else {
            if (large_tile) launch_fused_splits<cutlass::bfloat16_t,128>(x,w,out,splits);
            else launch_fused_splits<cutlass::bfloat16_t,64>(x,w,out,splits);
        }
        return out;
    }
    auto partial = splits == 1 ? out : at::empty({splits, M, N}, out.options());
    if (x.scalar_type() == at::kHalf) {
        if (tile == 16)
            launch_fixed<cutlass::half_t, 16>(x, w, partial, M, N, K, splits);
        else if (tile == 32)
            launch_fixed<cutlass::half_t, 32>(x, w, partial, M, N, K, splits);
        else if (tile == 64)
            launch_fixed<cutlass::half_t, 64>(x, w, partial, M, N, K, splits);
        else
            launch_fixed<cutlass::half_t, 128>(x, w, partial, M, N, K, splits);
    } else {
        if (tile == 16)
            launch_fixed<cutlass::bfloat16_t, 16>(x, w, partial, M, N, K, splits);
        else if (tile == 32)
            launch_fixed<cutlass::bfloat16_t, 32>(x, w, partial, M, N, K, splits);
        else if (tile == 64)
            launch_fixed<cutlass::bfloat16_t, 64>(x, w, partial, M, N, K, splits);
        else
            launch_fixed<cutlass::bfloat16_t, 128>(x, w, partial, M, N, K, splits);
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

Tensor fixed_projection(Tensor x, Tensor w, int64_t splits, int64_t tile) {
    return fixed_projection_impl(x,w,splits,tile,Tensor{});
}
Tensor fixed_projection_out(Tensor x, Tensor w, int64_t splits, int64_t tile, Tensor out) {
    return fixed_projection_impl(x,w,splits,tile,out);
}

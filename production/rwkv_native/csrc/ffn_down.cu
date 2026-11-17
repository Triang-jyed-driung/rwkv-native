// Fixed-order split sums with direct dual-layout loads and a residual epilogue.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"
using at::Tensor;

#include <torch/extension.h>
#include <type_traits>
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
template<class P,int MT,bool ROW,bool RES>
void launch_fused_splits(Tensor x,Tensor w,Tensor out,Tensor residual,int splits) {
    using Row=cutlass::layout::RowMajor;using Col=cutlass::layout::ColumnMajor;using BLayout=std::conditional_t<ROW,Row,Col>;
    using Op=cutlass::epilogue::thread::LinearCombination<float,4,float,float,RES?cutlass::epilogue::thread::ScaleType::NoBetaScaling:cutlass::epilogue::thread::ScaleType::Nothing>;
    using Gemm=cutlass::gemm::device::GemmBatched<P,Row,P,BLayout,float,Row,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<MT,128,64>,cutlass::gemm::GemmShape<32,32,64>,cutlass::gemm::GemmShape<16,8,16>,Op,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,3,8,8>;
    using Kernel=typename Gemm::GemmKernel;
    int M=x.size(0),K=x.size(1),N=w.size(1);
    cutlass::gemm::GemmCoord grid((M+MT-1)/MT,(N+127)/128,1);
    typename Kernel::Params p({M,N,K/splits},grid,{reinterpret_cast<P*>(x.data_ptr()),Row(K)},K/splits,
        {reinterpret_cast<P*>(w.data_ptr()),BLayout(ROW?N:K)},ROW?int64_t(K/splits)*N:K/splits,{RES?residual.data_ptr<float>():nullptr,Row(N)},0,{out.data_ptr<float>(),Row(N)},0,{1.f,1.f},1);
    int smem=sizeof(typename Kernel::SharedStorage);
    if(smem>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(fused_splits<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    fused_splits<Kernel><<<dim3(grid.m(),grid.n()),Kernel::kThreadCount,smem,c10::cuda::getCurrentCUDAStream()>>>(p,splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


Tensor run(Tensor x,Tensor w,Tensor residual,bool add) {
 TORCH_CHECK(x.is_cuda()&&w.device()==x.device()&&x.dim()==2&&w.dim()==2&&x.is_contiguous(),"device/shape");
 int M=x.size(0),K=x.size(1),N=w.size(1);
 TORCH_CHECK(w.size(0)==K&&K==4*N&&(N==2560||N==4096)&&M>=768,"dimensions");
 TORCH_CHECK(w.is_contiguous()||w.transpose(0,1).is_contiguous(),"layout");
 TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0&&reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"unaligned operand");
 TORCH_CHECK(x.scalar_type()==w.scalar_type()&&(x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16),"dtype");
 if(add)TORCH_CHECK(residual.device()==x.device()&&residual.scalar_type()==at::kFloat&&residual.sizes()==at::IntArrayRef({M,N})&&residual.is_contiguous(),"residual");
 c10::cuda::CUDAGuard guard(x.device());auto out=at::empty({M,N},x.options().dtype(at::kFloat));
 #define CALL(P,R,S) launch_fused_splits<P,128,R,S>(x,w,out,residual,8)
 #define TYPE(P) if(w.is_contiguous()){if(add){CALL(P,true,true);}else{CALL(P,true,false);}}else{if(add){CALL(P,false,true);}else{CALL(P,false,false);}}
 if(x.scalar_type()==at::kHalf){TYPE(cutlass::half_t);}else{TYPE(cutlass::bfloat16_t);}
 return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

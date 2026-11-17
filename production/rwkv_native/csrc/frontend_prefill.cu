// Experimental residual/norm/token-shift fusion, preserving the 256-thread tree.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
using at::Tensor;
__device__ __forceinline__ float row_sum(float value,float* scratch) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);
    if(!lane)scratch[warp]=value;
    __syncthreads();
    value=threadIdx.x<8?scratch[threadIdx.x]:0.f;
    if(!warp){for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);if(!lane)scratch[8]=value;}
    __syncthreads();return scratch[8];
}
template<class P,int C,bool SAVE>
__global__ __launch_bounds__(256,2) void fused_frontend(
    const float* x,const float* product,const float* previous,const P* weight,const P* bias,
    const P* coefficient,P* packed,float* normalized,float* state,float* residual,int T) {
    const int row=blockIdx.x,t=row%T,b=row/T;
    const int64_t base=int64_t(row)*C;
    __shared__ float scratch[36];
    float current[C/256],past[C/256];
    float sum=0.f,oldsum=0.f;
    #pragma unroll
    for(int j=0;j<C/256;++j){
        int c=threadIdx.x+j*256;
        current[j]=__fadd_rn(x[base+c],product[base+c]);sum+=current[j];
        if(t){past[j]=__fadd_rn(x[base-C+c],product[base-C+c]);oldsum+=past[j];}
    }
    float mean=row_sum(sum,scratch)/C,oldmean=t?row_sum(oldsum,scratch+18)/C:0.f;
    sum=0.f;oldsum=0.f;
    #pragma unroll
    for(int j=0;j<C/256;++j){
        float d=current[j]-mean;sum=fmaf(d,d,sum);
        if(t){float oldd=past[j]-oldmean;oldsum=fmaf(oldd,oldd,oldsum);}
    }
    float inverse=rsqrtf(row_sum(sum,scratch+9)/C+1e-5f);
    float oldinverse=t?rsqrtf(row_sum(oldsum,scratch+27)/C+1e-5f):0.f;
    #pragma unroll
    for(int j=0;j<C/256;++j){
        int c=threadIdx.x+j*256;int64_t i=base+c;
        float scale=float(weight[c]),shift=float(bias[c]);
        float z=__fadd_rn(__fmul_rn((current[j]-mean)*inverse,scale),shift);
        float old=t?__fadd_rn(__fmul_rn((past[j]-oldmean)*oldinverse,scale),shift):previous[int64_t(b)*C+c];
        packed[i]=P(__fadd_rn(z,__fmul_rn(old-z,float(coefficient[c]))));
        residual[i]=current[j];
        if constexpr(SAVE)normalized[i]=z;
        if(t==T-1)state[int64_t(b)*C+c]=z;
    }
}
std::vector<Tensor> run(std::vector<Tensor> xs,Tensor mask,bool save){
    TORCH_CHECK(xs.size()==6,"inputs");auto x=xs[0];
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kFloat&&x.dim()==3&&x.is_contiguous(),"x");
    int B=x.size(0),T=x.size(1),C=x.size(2);
    TORCH_CHECK(mask.device()==x.device()&&mask.scalar_type()==at::kBool&&mask.sizes()==at::IntArrayRef({B,T})&&mask.is_contiguous(),"mask metadata");
    TORCH_CHECK(T>1&&(C==2560||C==4096),"shape");
    TORCH_CHECK(xs[1].sizes()==at::IntArrayRef({B*T,C})&&xs[2].sizes()==at::IntArrayRef({B,C}),"state/projection");
    auto dtype=xs[3].scalar_type();TORCH_CHECK(dtype==at::kHalf||dtype==at::kBFloat16,"dtype");
    for(int i=0;i<6;++i){TORCH_CHECK(xs[i].device()==x.device()&&xs[i].is_contiguous(),"layout/device");TORCH_CHECK(xs[i].scalar_type()==(i<3?at::kFloat:dtype),"dtype mismatch");if(i>=3)TORCH_CHECK(xs[i].numel()==C,"parameter shape");}
    c10::cuda::CUDAGuard guard(x.device());
    auto packed=at::empty({B*T,C},xs[3].options()),normalized=at::empty(save?x.sizes():at::IntArrayRef({0}),x.options());
    auto state=at::empty_like(xs[2]),residual=at::empty_like(x);
    #define CALL(N,S) fused_frontend<scalar_t,N,S><<<B*T,256,0,c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),xs[1].data_ptr<float>(),xs[2].data_ptr<float>(),xs[3].data_ptr<scalar_t>(),xs[4].data_ptr<scalar_t>(),xs[5].data_ptr<scalar_t>(),packed.data_ptr<scalar_t>(),save?normalized.data_ptr<float>():nullptr,state.data_ptr<float>(),residual.data_ptr<float>(),T)
    if(B)AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype,"fused_frontend",[&]{if(C==2560){if(save){CALL(2560,true);}else{CALL(2560,false);}}else{if(save){CALL(4096,true);}else{CALL(4096,false);}}});
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {packed,normalized,state,residual};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

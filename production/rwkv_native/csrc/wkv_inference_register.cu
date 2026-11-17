// Native key/value preparation fused before the unchanged register WKV loop.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
using at::Tensor;
constexpr int D=64;
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for(int offset=16;offset;offset>>=1)x+=__shfl_down_sync(0xffffffff,x,offset);
    return __shfl_sync(0xffffffff,x,0);
}
#include "wkv_register_interleaved.cuh"
template<class P>
__global__ void prepare_rows(const float* w,const float* k,const float* v,const float* raw_rate,
                            const float* raw_value,const float* first,const P* w0,const P* scale,
                            const P* a0,const P* adapt,const P* v0,float* decay,float* key,
                            float* negative,float* adapted,float* value,int64_t heads,int H,bool blend) {
    const int lane=threadIdx.x&31;
    const int64_t head=int64_t(blockIdx.x)*4+threadIdx.x/32;
    if(head>=heads)return;
    const int column=(head%H)*64;
    float kk[2],u[2],rate[2],sum=0.f;
#pragma unroll
    for(int j=0;j<2;++j) {
        const int c=column+lane+j*32;const int64_t i=head*64+lane+j*32;
        kk[j]=k[i];u[j]=kk[j]*float(scale[c]);
        rate[j]=1.f/(1.f+expf(-(raw_rate[i]+float(a0[c]))));
        sum+=u[j]*u[j];
    }
    const float inverse=1.f/fmaxf(sqrtf(warp_sum(sum)),1e-12f);
#pragma unroll
    for(int j=0;j<2;++j) {
        const int c=column+lane+j*32;const int64_t i=head*64+lane+j*32;
        const float q=u[j]*inverse;
        key[i]=kk[j]*(1.f+(rate[j]-1.f)*float(adapt[c]));
        negative[i]=-q;adapted[i]=q*rate[j];
        const float ww=__fadd_rn(w[i],float(w0[c]));
        decay[i]=expf(-0.6065306597126334f/(1.f+expf(-ww)));
        float vv=v[i];
        if(blend) {
            const float gate=1.f/(1.f+expf(-__fadd_rn(raw_value[i],float(v0[c]))));
            vv=__fadd_rn(vv,__fmul_rn(first[i]-vv,gate));
        }
        value[i]=vv;
    }
}
std::vector<Tensor> wkv_register_prepared(std::vector<Tensor> xs,Tensor initial,Tensor mask,std::vector<Tensor> params,Tensor first,bool blend) {
    TORCH_CHECK(xs.size()==6 && params.size()==5,"expected six vectors and five parameters");
    auto r=xs[0];TORCH_CHECK(r.is_cuda() && r.dim()==4 && r.size(3)==64,"input shape/device");
    c10::cuda::CUDAGuard guard(r.device());
    int B=r.size(0),T=r.size(1),H=r.size(2);
    TORCH_CHECK(B*H>=32 && B*H<128 && T>=256,"register schedule scope");
    for(const auto& x:xs)TORCH_CHECK(x.device()==r.device() && x.scalar_type()==at::kFloat && x.sizes()==r.sizes() && x.is_contiguous(),"vector");
    TORCH_CHECK(initial.device()==r.device() && initial.scalar_type()==at::kFloat && initial.sizes()==at::IntArrayRef({B,H,64,64}),"initial");
    TORCH_CHECK(mask.device()==r.device() && mask.scalar_type()==at::kBool && mask.is_contiguous() && mask.sizes()==at::IntArrayRef({B,T}),"mask");
    TORCH_CHECK(first.device()==r.device() && first.scalar_type()==at::kFloat && first.sizes()==r.sizes() && first.is_contiguous(),"first");
    auto dtype=params[0].scalar_type();TORCH_CHECK(dtype==at::kHalf || dtype==at::kBFloat16,"dtype");
    for(const auto& p:params)TORCH_CHECK(p.device()==r.device() && p.scalar_type()==dtype && p.numel()==H*64 && p.is_contiguous(),"parameter");
    initial=initial.contiguous();auto final=at::empty_like(initial),y=at::empty_like(r);
    auto decay=at::empty_like(r),key=at::empty_like(r),negative=at::empty_like(r),adapted=at::empty_like(r),value=at::empty_like(r);
    auto stream=c10::cuda::getCurrentCUDAStream();
#define PREP(P) prepare_rows<<<(B*T*H+3)/4,128,0,stream>>>(xs[1].data_ptr<float>(),xs[2].data_ptr<float>(),xs[3].data_ptr<float>(),xs[4].data_ptr<float>(),xs[5].data_ptr<float>(),first.data_ptr<float>(),params[0].data_ptr<P>(),params[1].data_ptr<P>(),params[2].data_ptr<P>(),params[3].data_ptr<P>(),params[4].data_ptr<P>(),decay.data_ptr<float>(),key.data_ptr<float>(),negative.data_ptr<float>(),adapted.data_ptr<float>(),value.data_ptr<float>(),int64_t(B)*T*H,H,blend)
    if(dtype==at::kHalf){PREP(at::Half);}else{PREP(at::BFloat16);}
#undef PREP
#define WKV(P) forward_register_interleaved<float,false,2,P><<<dim3(B*H,8),128,0,stream>>>(r.data_ptr<float>(),nullptr,key.data_ptr<float>(),value.data_ptr<float>(),negative.data_ptr<float>(),adapted.data_ptr<float>(),initial.data_ptr<float>(),mask.data_ptr<bool>(),y.data_ptr<float>(),final.data_ptr<float>(),nullptr,decay.data_ptr<float>(),T,H,16,0)
    if(B*H<64){WKV(8);}else{WKV(4);}
#undef WKV
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,final,key,value};
}

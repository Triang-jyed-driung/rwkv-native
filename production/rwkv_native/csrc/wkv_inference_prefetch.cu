// Experimental inference fusion; keep FP32 recurrence and readout reductions.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include "wkv_inference_prefetch.cuh"
using at::Tensor;
std::vector<Tensor> wkv_prefetched_readout(std::vector<Tensor> xs,Tensor initial,Tensor mask,Tensor gate,
                        std::vector<Tensor> params,Tensor first_value,bool blend) {
    TORCH_CHECK(xs.size()==6 && params.size()==8,"six vectors and eight parameters");
    const auto r=xs[0];
    TORCH_CHECK(r.is_cuda() && r.dim()==4 && r.size(3)==64 && r.numel()>0,"r shape/device");
    c10::cuda::CUDAGuard guard(r.device());
    for(const auto& x:xs)
        TORCH_CHECK(x.device()==r.device() && x.scalar_type()==at::kFloat && x.sizes()==r.sizes() &&
                    x.is_contiguous() && reinterpret_cast<uintptr_t>(x.data_ptr())%16==0,"vectors");
    int B=r.size(0),T=r.size(1),H=r.size(2);
    TORCH_CHECK(initial.device()==r.device() && initial.scalar_type()==at::kFloat &&
                initial.sizes()==at::IntArrayRef({B,H,64,64}) && initial.stride(0)>=0 &&
                initial.stride(1)==4096 && initial.stride(2)==64 && initial.stride(3)==1,"initial");
    TORCH_CHECK(mask.device()==r.device() && mask.scalar_type()==at::kBool && mask.is_contiguous() &&
                mask.sizes()==at::IntArrayRef({B,T}),"mask");
    TORCH_CHECK(gate.device()==r.device() && gate.scalar_type()==at::kFloat && gate.is_contiguous() &&
                gate.numel()==r.numel() && reinterpret_cast<uintptr_t>(gate.data_ptr())%16==0,"gate");
    const auto dtype=params[0].scalar_type();
    TORCH_CHECK(dtype==at::kHalf || dtype==at::kBFloat16,"parameter dtype");
    for(const auto& p:params)
        TORCH_CHECK(p.device()==r.device() && p.scalar_type()==dtype && p.is_contiguous() && p.numel()==H*64,"parameters");
    TORCH_CHECK(first_value.device()==r.device() && first_value.scalar_type()==at::kFloat &&
                first_value.is_contiguous() && first_value.numel()==r.numel(),"first_value");
    auto y=at::empty(r.sizes(),r.options().dtype(dtype));
    auto final=at::empty({B,H,64,64},initial.options());
    const auto stream=c10::cuda::getCurrentCUDAStream();
#define RUN(P,F) fused_wkv_prepare_readout_float<P,F,false><<<B*H,64,0,stream>>>( \
    xs[0].data_ptr<float>(),xs[1].data_ptr<float>(),xs[2].data_ptr<float>(),xs[3].data_ptr<float>(), \
    xs[4].data_ptr<float>(),xs[5].data_ptr<float>(),initial.data_ptr<float>(),mask.data_ptr<bool>(), \
    y.data_ptr<P>(),final.data_ptr<float>(),nullptr,T,H,16,0,initial.stride(0),gate.data_ptr<float>(), \
    params[0].data_ptr<P>(),params[1].data_ptr<P>(),params[2].data_ptr<P>(),first_value.data_ptr<float>(), \
    params[3].data_ptr<P>(),params[4].data_ptr<P>(),params[5].data_ptr<P>(),params[6].data_ptr<P>(),params[7].data_ptr<P>(),blend)
#define DISPATCH(P) if(T==1){RUN(P,1);}else if(T<32){RUN(P,2);}else{RUN(P,4);}
    if(dtype==at::kHalf){DISPATCH(at::Half);}else{DISPATCH(at::BFloat16);}
#undef DISPATCH
#undef RUN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y,final};
}

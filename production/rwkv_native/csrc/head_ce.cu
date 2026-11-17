// Recomputed vocabulary logits become the CE derivative in their own buffer.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
using at::Tensor;

template<bool MAXIMUM, int THREADS>
__device__ float block_reduce(float value) {
    __shared__ float warps[THREADS/32];
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    for(int offset=16;offset;offset/=2) {
        const float other=__shfl_down_sync(0xffffffff,value,offset);
        value=MAXIMUM?fmaxf(value,other):value+other;
    }
    if(!lane)warps[warp]=value;
    __syncthreads();
    if(!warp) {
        value=lane<THREADS/32?warps[lane]:(MAXIMUM?-INFINITY:0.f);
        for(int offset=16;offset;offset/=2) {
            const float other=__shfl_down_sync(0xffffffff,value,offset);
            value=MAXIMUM?fmaxf(value,other):value+other;
        }
        if(!lane)warps[0]=value;
    }
    __syncthreads();
    const float result=warps[0];
    __syncthreads();
    return result;
}

template<int THREADS>
__global__ void ce_gradient(float* logits,const int64_t* targets,const float* factor,int classes) {
    float* row=logits+int64_t(blockIdx.x)*classes;
    const int64_t target=targets[blockIdx.x];
    if(target==-100) {
        for(int j=threadIdx.x;j<classes;j+=THREADS)row[j]=0.f;
        return;
    }
    float maximum=-INFINITY;
    for(int j=threadIdx.x;j<classes;j+=THREADS)maximum=fmaxf(maximum,row[j]);
    maximum=block_reduce<true,THREADS>(maximum);
    float sum=0.f;
    for(int j=threadIdx.x;j<classes;j+=THREADS)sum+=expf(row[j]-maximum);
    const float inverse=1.f/block_reduce<false,THREADS>(sum);
    const float scale=factor[0];
    for(int j=threadIdx.x;j<classes;j+=THREADS)
        row[j]=(expf(row[j]-maximum)*inverse-float(j==target))*scale;
}

Tensor gradient_inplace(Tensor logits,Tensor targets,Tensor factor) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type()==at::kFloat && logits.dim()==2
                && logits.is_contiguous(),"expected contiguous FP32 logits");
    TORCH_CHECK(targets.device()==logits.device() && targets.scalar_type()==at::kLong
                && targets.is_contiguous() && targets.numel()==logits.size(0),"invalid targets");
    TORCH_CHECK(factor.device()==logits.device() && factor.scalar_type()==at::kFloat
                && factor.numel()==1 && factor.is_contiguous(),"expected a CUDA FP32 loss scale");
    TORCH_CHECK(logits.size(0)<=INT_MAX && logits.size(1)>0 && logits.size(1)<=INT_MAX,
                "invalid vocabulary dimensions");
    c10::cuda::CUDAGuard guard(logits.device());
    if(logits.size(0)) {
        auto stream=c10::cuda::getCurrentCUDAStream();
        if(logits.size(1)>=4096)
            ce_gradient<1024><<<logits.size(0),1024,0,stream>>>(logits.data_ptr<float>(),targets.data_ptr<int64_t>(),factor.data_ptr<float>(),logits.size(1));
        else
            ce_gradient<256><<<logits.size(0),256,0,stream>>>(logits.data_ptr<float>(),targets.data_ptr<int64_t>(),factor.data_ptr<float>(),logits.size(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("gradient_",&gradient_inplace);}

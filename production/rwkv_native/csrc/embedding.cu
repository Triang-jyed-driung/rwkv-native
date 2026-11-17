// Gather directly into FP32; accumulate repeated-token gradients in FP32
// registers and store parameter precision without a dense FP32 vocabulary buffer.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <climits>
using at::Tensor;

template<typename P>
__global__ void gather_fp32(const int64_t* tokens,const P* weights,float* out,int64_t count,int C,int V) {
    int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count*int64_t(C)) {
        int64_t row=tokens[index/C];
        CUDA_KERNEL_ASSERT(row>=0 && row<V);
        out[index]=float(weights[row*C+index%C]);
    }
}
__global__ void token_positions(int* positions,int count) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)positions[i]=int(i);
}
template<typename P>
__global__ void sum_token_gradients(const int64_t* sorted,const int* positions,const float* grad,
                                  P* output,int count,int C) {
    int start=blockIdx.x;
    int64_t token=sorted[start];
    if(start>0 && sorted[start-1]==token)return;
    int c=blockIdx.y*blockDim.x+threadIdx.x;
    if(c>=C)return;
    float sum=0.f;
    // Stable radix sorting preserves token order within each group.
    for(int i=start;i<count && sorted[i]==token;++i)
        sum+=grad[int64_t(positions[i])*C+c];
    output[token*C+c]=P(sum);
}
Tensor embedding_forward(Tensor tokens,Tensor weights) {
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type()==at::kLong && tokens.is_contiguous(),"expected contiguous CUDA int64 tokens");
    TORCH_CHECK(weights.device()==tokens.device() && weights.dim()==2 && weights.is_contiguous() &&
                (weights.scalar_type()==at::kHalf || weights.scalar_type()==at::kBFloat16),"expected FP16/BF16 embedding weights");
    TORCH_CHECK(weights.size(0)>0 && weights.size(0)<=INT_MAX && weights.size(1)>0 && weights.size(1)<=INT_MAX,"invalid embedding shape");
    c10::cuda::CUDAGuard guard(tokens.device());
    auto shape=tokens.sizes().vec();shape.push_back(weights.size(1));
    auto out=at::empty(shape,weights.options().dtype(at::kFloat));
    if(out.numel()) {
        auto stream=c10::cuda::getCurrentCUDAStream();
        if(weights.scalar_type()==at::kHalf)
            gather_fp32<<<(out.numel()+255)/256,256,0,stream>>>(tokens.data_ptr<int64_t>(),weights.data_ptr<at::Half>(),out.data_ptr<float>(),tokens.numel(),weights.size(1),weights.size(0));
        else
            gather_fp32<<<(out.numel()+255)/256,256,0,stream>>>(tokens.data_ptr<int64_t>(),weights.data_ptr<at::BFloat16>(),out.data_ptr<float>(),tokens.numel(),weights.size(1),weights.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
Tensor embedding_backward(Tensor grad,Tensor tokens,int64_t rows,bool bf16) {
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type()==at::kLong && tokens.is_contiguous(),"expected contiguous CUDA int64 tokens");
    TORCH_CHECK(grad.device()==tokens.device() && grad.is_contiguous() && grad.scalar_type()==at::kFloat &&
                grad.dim()==tokens.dim()+1 && grad.size(-1)>0 && grad.numel()/grad.size(-1)==tokens.numel(),"invalid FP32 embedding gradient");
    TORCH_CHECK(rows>0 && rows<=INT_MAX && tokens.numel()<=INT_MAX && grad.size(-1)<=INT_MAX,"embedding dimensions exceed index range");
    c10::cuda::CUDAGuard guard(tokens.device());
    int count=tokens.numel(),C=grad.size(-1);
    auto output=at::zeros({rows,int64_t(C)},grad.options().dtype(bf16?at::kBFloat16:at::kHalf));
    if(!count)return output;
    auto sorted=at::empty_like(tokens),positions=at::empty({count},tokens.options().dtype(at::kInt));
    auto ordered=at::empty_like(positions);
    auto stream=c10::cuda::getCurrentCUDAStream();
    token_positions<<<(int64_t(count)+255)/256,256,0,stream>>>(positions.data_ptr<int>(),count);
    int bits=1;for(int64_t high=rows-1;high>1;high>>=1)++bits;
    size_t bytes=0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,bytes,tokens.data_ptr<int64_t>(),sorted.data_ptr<int64_t>(),
                   positions.data_ptr<int>(),ordered.data_ptr<int>(),count,0,bits,stream));
    auto scratch=at::empty({int64_t(bytes)},tokens.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch.data_ptr(),bytes,tokens.data_ptr<int64_t>(),sorted.data_ptr<int64_t>(),
                   positions.data_ptr<int>(),ordered.data_ptr<int>(),count,0,bits,stream));
    dim3 grid(count,(C+255)/256);
    if(bf16)sum_token_gradients<<<grid,256,0,stream>>>(sorted.data_ptr<int64_t>(),ordered.data_ptr<int>(),grad.data_ptr<float>(),output.data_ptr<at::BFloat16>(),count,C);
    else sum_token_gradients<<<grid,256,0,stream>>>(sorted.data_ptr<int64_t>(),ordered.data_ptr<int>(),grad.data_ptr<float>(),output.data_ptr<at::Half>(),count,C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward",&embedding_forward);
    m.def("backward",&embedding_backward);
}

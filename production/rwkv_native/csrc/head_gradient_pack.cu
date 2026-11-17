// Single FP16 component for the scaled BF16 head adjoint. The caller already
// holds the FP16 activation operand: do not replicate it or generate unused lows.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
using at::Tensor;
__device__ float maximum(float a,float b){return __uint_as_float(max(__float_as_uint(a),__float_as_uint(b)));}
__device__ float block_max(float v){
 __shared__ unsigned warp[8];unsigned bits=__reduce_max_sync(0xffffffff,__float_as_uint(v));
 if(threadIdx.x%32==0)warp[threadIdx.x/32]=bits;__syncthreads();
 bits=threadIdx.x<8?warp[threadIdx.x]:0;
 if(threadIdx.x<32)bits=__reduce_max_sync(0xffffffff,bits);
 return __uint_as_float(bits);
}
__global__ void ranges(const float*g,float*partial,int64_t n){
 float v=0;for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<n;i+=int64_t(gridDim.x)*256)v=maximum(v,fabsf(g[i]));
 v=block_max(v);if(!threadIdx.x)partial[blockIdx.x]=v;
}
__global__ void factors(const float*partial,const float*range,float*scale,float*coeff,float*dx,int n,bool add){
 float v=0;for(int i=threadIdx.x;i<n;i+=256)v=maximum(v,partial[i]);v=block_max(v);
 if(!threadIdx.x){
  unsigned bits=__float_as_uint(v)&0x7fffffffu,field=bits>>23;
  int exponent=(field&&field<255)?int(field)-126:(!field&&bits?31-__clz(bits)-148:0);
  int shift=max(-120,min(120,7-exponent));
  scale[0]=__uint_as_float(unsigned(127+shift)<<23);
  coeff[0]=__fmul_rn(__uint_as_float(unsigned(127-shift)<<23),range[3]);coeff[1]=add?1.f:0.f;
  dx[0]=__fdiv_rn(range[5],scale[0]);
 }
}
__global__ void pack_values(const float*g,const float*scale,at::Half*out,int64_t n){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<n;i+=int64_t(gridDim.x)*256)out[i]=at::Half(__fmul_rn(g[i],scale[0]));
}
__global__ void copy_scaled(const float*x,const float*scale,float*out,int64_t n){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<n;i+=int64_t(gridDim.x)*256)out[i]=__fmul_rn(x[i],scale[0]);
}
std::vector<Tensor> pack(Tensor g,Tensor range,bool add){
 TORCH_CHECK(g.is_cuda()&&g.scalar_type()==at::kFloat&&g.is_contiguous()&&g.dim()==2&&g.numel()>0,"nonempty FP32 gradient");
 TORCH_CHECK(range.device()==g.device()&&range.scalar_type()==at::kFloat&&range.is_contiguous()&&range.numel()==9,"range factors");
 c10::cuda::CUDAGuard guard(g.device());auto stream=c10::cuda::getCurrentCUDAStream();
 int blocks=std::min<int64_t>((g.numel()+255)/256,1024);
 auto scratch=at::empty({blocks},g.options()),scale=at::empty({},g.options()),coeff=at::empty({2},g.options()),dx=at::empty({},g.options());
 auto out=at::empty_like(g,g.options().dtype(at::kHalf));
 ranges<<<blocks,256,0,stream>>>(g.data_ptr<float>(),scratch.data_ptr<float>(),g.numel());
 factors<<<1,256,0,stream>>>(scratch.data_ptr<float>(),range.data_ptr<float>(),scale.data_ptr<float>(),coeff.data_ptr<float>(),dx.data_ptr<float>(),blocks,add);
 blocks=std::min<int64_t>((g.numel()+255)/256,4096);
 pack_values<<<blocks,256,0,stream>>>(g.data_ptr<float>(),scale.data_ptr<float>(),out.data_ptr<at::Half>(),g.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,coeff,dx};
}
void unpack(Tensor x,Tensor scale,Tensor out){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kFloat&&x.is_contiguous(),"FP32 product");
 TORCH_CHECK(out.device()==x.device()&&out.scalar_type()==at::kFloat&&out.sizes()==x.sizes()&&out.is_contiguous(),"FP32 destination");
 TORCH_CHECK(scale.device()==x.device()&&scale.scalar_type()==at::kFloat&&scale.numel()==1&&scale.is_contiguous(),"scale");
 c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream();
 if(x.numel())copy_scaled<<<std::min<int64_t>((x.numel()+255)/256,4096),256,0,stream>>>(x.data_ptr<float>(),scale.data_ptr<float>(),out.data_ptr<float>(),x.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pack",&pack);m.def("unpack",&unpack);}

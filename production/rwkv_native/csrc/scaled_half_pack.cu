// Scaled FP16 operands, FP32 GEMM accumulation. Range scaling follows wkv7_fast's
// power-of-two approach; all three operands are scaled independently.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
using at::Tensor;
__device__ float block_max(float x){
 __shared__ unsigned warp[8];unsigned bits=__reduce_max_sync(0xffffffff,__float_as_uint(x));
 if(threadIdx.x%32==0)warp[threadIdx.x/32]=bits;__syncthreads();
 bits=threadIdx.x<8?warp[threadIdx.x]:0;
 if(threadIdx.x<32)bits=__reduce_max_sync(0xffffffff,bits);
 return __uint_as_float(bits);
}
__global__ void ranges(const at::BFloat16*x,const float*g,const at::BFloat16*w,float*partial,int64_t nx,int64_t ng,int64_t nw){
 float vals[3]={};int64_t n=max(nx,max(ng,nw));
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<n;i+=int64_t(gridDim.x)*256){
  if(i<nx)vals[0]=fmaxf(vals[0],fabsf(float(x[i])));
  if(i<ng)vals[1]=fmaxf(vals[1],fabsf(g[i]));
  if(i<nw)vals[2]=fmaxf(vals[2],fabsf(float(w[i])));
 }
 #pragma unroll
 for(int q=0;q<3;q++){float v=block_max(vals[q]);if(threadIdx.x==0)partial[q*gridDim.x+blockIdx.x]=v;__syncthreads();}
}
__global__ void factors(const float*partial,float*out,int blocks){
 __shared__ int shifts[3];
 #pragma unroll
 for(int q=0;q<3;q++){
  float v=0.f;for(int i=threadIdx.x;i<blocks;i+=256)v=fmaxf(v,partial[q*blocks+i]);v=block_max(v);
  if(threadIdx.x==0){
   unsigned bits=__float_as_uint(v)&0x7fffffffu,field=bits>>23;
   int exponent=(field && field<255)?int(field)-126:(!field && bits?31-__clz(bits)-148:0);
   int shift=max(-120,min(120,7-exponent));shifts[q]=shift;
   out[q]=ldexpf(1.f,shift);out[q+3]=ldexpf(1.f,-shift);
  }
  __syncthreads();
 }
 if(threadIdx.x==0){out[6]=ldexpf(1.f,-shifts[0]-shifts[1]);out[7]=0.f;out[8]=ldexpf(1.f,-shifts[1]-shifts[2]);}
}
__global__ void pack_values(const at::BFloat16*x,const float*g,const at::BFloat16*w,const float*scale,
 at::Half*xh,at::Half*gh,at::Half*wh,int64_t nx,int64_t ng,int64_t nw){
 for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<max(nx,max(ng,nw));i+=int64_t(gridDim.x)*256){
  if(i<nx)xh[i]=at::Half(__fmul_rn(float(x[i]),scale[0]));
  if(i<ng)gh[i]=at::Half(__fmul_rn(g[i],scale[1]));
  if(i<nw)wh[i]=at::Half(__fmul_rn(float(w[i]),scale[2]));
 }
}
std::vector<Tensor> pack(Tensor x,Tensor g,Tensor w){
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),"BF16 contiguous x");
 TORCH_CHECK(g.device()==x.device() && g.scalar_type()==at::kFloat && g.is_contiguous(),"FP32 contiguous gradient");
 TORCH_CHECK(w.device()==x.device() && w.scalar_type()==at::kBFloat16 && (w.is_contiguous() || (w.dim()==2 && w.transpose(0,1).is_contiguous())),"dense BF16 weight");
 c10::cuda::CUDAGuard guard(x.device());
 auto xh=at::empty_like(x,x.options().dtype(at::kHalf));auto gh=at::empty_like(g,g.options().dtype(at::kHalf));
 auto wh=at::empty_strided(w.sizes(),w.strides(),w.options().dtype(at::kHalf));
 int64_t n=std::max(x.numel(),std::max(g.numel(),w.numel()));TORCH_CHECK(n>0,"nonempty");int blocks=std::min<int64_t>((n+255)/256,4096);
 auto scratch=at::empty({3,blocks},g.options()),scale=at::empty({9},g.options());auto stream=c10::cuda::getCurrentCUDAStream();
 ranges<<<blocks,256,0,stream>>>(x.data_ptr<at::BFloat16>(),g.data_ptr<float>(),w.data_ptr<at::BFloat16>(),scratch.data_ptr<float>(),x.numel(),g.numel(),w.numel());
 factors<<<1,256,0,stream>>>(scratch.data_ptr<float>(),scale.data_ptr<float>(),blocks);
 pack_values<<<blocks,256,0,stream>>>(x.data_ptr<at::BFloat16>(),g.data_ptr<float>(),w.data_ptr<at::BFloat16>(),scale.data_ptr<float>(),xh.data_ptr<at::Half>(),gh.data_ptr<at::Half>(),wh.data_ptr<at::Half>(),x.numel(),g.numel(),w.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {xh,gh,wh,scale};
}
static void check(cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"cuBLAS status ",int(s));}
Tensor dw(Tensor x,Tensor g,Tensor coefficients,Tensor out){
 TORCH_CHECK(x.is_cuda() && x.dim()==2 && g.dim()==2 && x.is_contiguous() && g.is_contiguous() && x.scalar_type()==at::kHalf && g.scalar_type()==at::kHalf && g.device()==x.device() && x.size(0)==g.size(0),"FP16 matrices");
 TORCH_CHECK(coefficients.device()==x.device() && coefficients.scalar_type()==at::kFloat && coefficients.is_contiguous() && coefficients.numel()==2,"alpha/beta");
 c10::cuda::CUDAGuard guard(x.device());
 if(out.numel()==0)out=at::empty({x.size(1),g.size(1)},x.options().dtype(at::kFloat));
 TORCH_CHECK(out.device()==x.device() && out.scalar_type()==at::kFloat && out.is_contiguous() && out.sizes()==at::IntArrayRef({x.size(1),g.size(1)}),"FP32 gradient output");
 auto handle=at::cuda::getCurrentCUDABlasHandle();cublasPointerMode_t oldp;cublasMath_t oldm;
 check(cublasGetPointerMode(handle,&oldp));check(cublasGetMathMode(handle,&oldm));
 check(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
 auto status=cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);
 if(status!=CUBLAS_STATUS_SUCCESS){cublasSetMathMode(handle,oldm);check(status);}
 status=cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,g.size(1),x.size(1),x.size(0),coefficients.data_ptr<float>(),g.data_ptr(),CUDA_R_16F,g.size(1),x.data_ptr(),CUDA_R_16F,x.size(1),coefficients.data_ptr<float>()+1,out.data_ptr<float>(),CUDA_R_32F,g.size(1),CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
 auto restorep=cublasSetPointerMode(handle,oldp);auto restorem=cublasSetMathMode(handle,oldm);
 check(status);check(restorep);check(restorem);return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pack",&pack);m.def("dw",&dw);}

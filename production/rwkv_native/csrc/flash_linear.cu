// CUDA translation of FlashRWKV2 _internal/linear.py, MIT.
// Canonical increasing-K m16n8k16 MMA, FullCol warp placement and two-stage copies.
// M blocks are adjacent in launch order to reuse a weight tile across rows.
// Both dense weight layouts are loaded directly; no runtime transpose buffer.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using at::Tensor;
__device__ __forceinline__ void copy16(void* dest,const void* src,bool valid) {
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dest));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(src),"r"(valid?16:0));
}
__device__ __forceinline__ void ldm4(const void* src,void* dest) {
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(src));auto p=(unsigned*)dest;
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(p[0]),"=r"(p[1]),"=r"(p[2]),"=r"(p[3]):"r"(address));
}
__device__ __forceinline__ void ldm4_trans(const void* src,void* dest) {
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(src));auto p=(unsigned*)dest;
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(p[0]),"=r"(p[1]),"=r"(p[2]),"=r"(p[3]):"r"(address));
}
template<class P> __device__ __forceinline__ void mma(float* d,const P* a,const P* b) {
    auto ap=(const unsigned*)a;auto bp=(const unsigned*)b;
    if constexpr(std::is_same_v<P,__half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(ap[0]),"r"(ap[1]),"r"(ap[2]),"r"(ap[3]),"r"(bp[0]),"r"(bp[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(ap[0]),"r"(ap[1]),"r"(ap[2]),"r"(ap[3]),"r"(bp[0]),"r"(bp[1]));
    }
}
template<int COPYK> __device__ __forceinline__ int index(int row,int col) {
    return row*COPYK+((col/8)^(row&(COPYK/8<8?COPYK/8-1:7)))*8;
}
template<class P,int MT,int NT,int COPYK,bool COL>
__device__ __forceinline__ void load_tiles(P* sa,P* sb,const P* x,const P* w,int M,int N,int K,int block) {
    #pragma unroll
    for(int i=threadIdx.x;i<MT*COPYK/8;i+=128) {
        int row=i/(COPYK/8),col=i%(COPYK/8)*8,global_row=blockIdx.x*MT+row;
        copy16(sa+index<COPYK>(row,col),x+int64_t(global_row)*K+block*COPYK+col,global_row<M);
    }
    #pragma unroll
    for(int i=threadIdx.x;i<NT*COPYK/8;i+=128) {
        if constexpr(COL) {
            int row=i/(NT/8),col=i%(NT/8)*8;
            copy16(sb+index<NT>(row,col),w+int64_t(block*COPYK+row)*N+blockIdx.y*NT+col,true);
        } else {
            int row=i/(COPYK/8),col=i%(COPYK/8)*8,global_row=blockIdx.y*NT+row;
            copy16(sb+index<COPYK>(row,col),w+int64_t(global_row)*K+block*COPYK+col,true);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template<class P,int MT,int NT,int COPYK,bool RELU,bool COL>
__global__ __launch_bounds__(128,1) void flash_linear_port(const P* __restrict__ x,const P* __restrict__ w,
                                                         void* __restrict__ output,const float* __restrict__ residual,int M,int N,int K) {
    extern __shared__ __align__(16) unsigned char shared[];
    P* sa=(P*)shared;P* sb=sa+2*MT*COPYK;
    float accum[MT/16][NT/32][4];
    #pragma unroll
    for(int m=0;m<MT/16;++m) {
        #pragma unroll
        for(int n=0;n<NT/32;++n) {
            #pragma unroll
            for(int q=0;q<4;++q)accum[m][n][q]=0.f;
        }
    }
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    load_tiles<P,MT,NT,COPYK,COL>(sa,sb,x,w,M,N,K,0);
    for(int block=0;block<K/COPYK;++block) {
        __syncthreads();
        if(block+1<K/COPYK) {
            load_tiles<P,MT,NT,COPYK,COL>(sa+((block+1)&1)*MT*COPYK,sb+((block+1)&1)*NT*COPYK,x,w,M,N,K,block+1);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        #pragma unroll
        for(int step=0;step<COPYK/16;++step) {
            P alocal[MT/16][8],blocal[NT/64][8];
            #pragma unroll
            for(int m=0;m<MT/16;++m)
                ldm4(sa+(block&1)*MT*COPYK+index<COPYK>(m*16+lane%16,step*16+(lane/16)*8),alocal[m]);
            #pragma unroll
            for(int n=0;n<NT/64;++n) {
                if constexpr(COL)
                    ldm4_trans(sb+(block&1)*NT*COPYK+index<NT>(step*16+((lane/8)%2)*8+lane%8,warp*(NT/4)+n*16+(lane/16)*8),blocal[n]);
                else ldm4(sb+(block&1)*NT*COPYK+index<COPYK>(warp*(NT/4)+n*16+(lane/16)*8+lane%8,step*16+((lane/8)%2)*8),blocal[n]);
            }
            #pragma unroll
            for(int m=0;m<MT/16;++m) {
                #pragma unroll
                for(int n=0;n<NT/64;++n) {
                    mma<P>(accum[m][n*2],alocal[m],blocal[n]);
                    mma<P>(accum[m][n*2+1],alocal[m],blocal[n]+4);
                }
            }
        }
    }
    #pragma unroll
    for(int m=0;m<MT/16;++m) {
        #pragma unroll
        for(int n=0;n<NT/32;++n) {
            #pragma unroll
            for(int q=0;q<4;++q) {
                const int row=blockIdx.x*MT+m*16+lane/4+(q/2)*8;
                const int col=blockIdx.y*NT+warp*(NT/4)+n*8+(lane%4)*2+q%2;
                if(row<M) {
                    const float v=accum[m][n][q];
                    if constexpr(RELU) {
                        if constexpr(std::is_same_v<P,__half>) ((P*)output)[int64_t(row)*N+col]=__float2half_rn(v>0.f?v*v:0.f);
                        else ((P*)output)[int64_t(row)*N+col]=__float2bfloat16_rn(v>0.f?v*v:0.f);
                    }
                    else ((float*)output)[int64_t(row)*N+col]=residual?v+residual[int64_t(row)*N+col]:v;
                }
            }
        }
    }
}
template<class P,int MT,int NT,int COPYK,bool RELU,bool COL> void launch(Tensor x,Tensor w,Tensor y,Tensor residual) {
    int smem=2*(MT+NT)*COPYK*sizeof(P);
    auto fn=flash_linear_port<P,MT,NT,COPYK,RELU,COL>;
    if(smem>=48*1024)C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    fn<<<dim3((x.size(0)+MT-1)/MT,w.size(0)/NT),128,smem,c10::cuda::getCurrentCUDAStream()>>>((P*)x.data_ptr(),(P*)w.data_ptr(),y.data_ptr(),residual.defined()?residual.data_ptr<float>():nullptr,x.size(0),w.size(0),x.size(1));
}
Tensor run_impl(Tensor x,Tensor w,int tile,bool relu,Tensor residual) {
    TORCH_CHECK(x.is_cuda() && w.device()==x.device() && x.is_contiguous() && (w.is_contiguous() || w.transpose(0,1).is_contiguous()) && x.dim()==2 && w.dim()==2 && x.size(1)==w.size(1),"shape/layout");
    TORCH_CHECK(x.scalar_type()==w.scalar_type() && (x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16),"dtype");
    TORCH_CHECK(x.size(1)>0 && w.size(0)>0 && x.size(1)%256==0 && w.size(0)%128==0,"alignment");
    TORCH_CHECK(tile==16 || tile==64 || tile==128 || tile==132 || tile==64128,"unsupported tile");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
                reinterpret_cast<uintptr_t>(w.data_ptr())%16==0,"unaligned operands");
    if(residual.defined())TORCH_CHECK(!relu && residual.device()==x.device() &&
        residual.scalar_type()==at::kFloat && residual.is_contiguous() &&
        residual.sizes()==at::IntArrayRef({x.size(0),w.size(0)}),"invalid FP32 residual");
    c10::cuda::CUDAGuard guard(x.device());
    auto y=at::empty({x.size(0),w.size(0)},x.options().dtype(relu?x.scalar_type():at::kFloat));
    if(!x.size(0))return y;
    #define CALL_LAYOUT(P,R,C) if(tile==64128)launch<P,64,128,64,R,C>(x,w,y,residual);else if(tile==16)launch<P,16,64,256,R,C>(x,w,y,residual);else if(tile==64)launch<P,64,64,64,R,C>(x,w,y,residual);else if(tile==132)launch<P,128,128,32,R,C>(x,w,y,residual);else launch<P,128,128,64,R,C>(x,w,y,residual)
    #define CALL(P,R) if(w.is_contiguous()){CALL_LAYOUT(P,R,false);}else{CALL_LAYOUT(P,R,true);}
    if(x.scalar_type()==at::kHalf){if(relu){CALL(__half,true);}else{CALL(__half,false);}}
    else {if(relu){CALL(__nv_bfloat16,true);}else{CALL(__nv_bfloat16,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
Tensor run(Tensor x,Tensor w,int tile,bool relu){return run_impl(x,w,tile,relu,Tensor());}
Tensor run_residual(Tensor x,Tensor w,Tensor residual,int tile){return run_impl(x,w,tile,false,residual);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("run_residual",&run_residual);}

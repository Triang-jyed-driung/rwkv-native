// Experimental single-row grouped projection, SM80+.
// Uses the canonical MMA order and cp.async helpers of this project's
// flash_linear.cu (CUDA translation of MIT FlashRWKV2 linear.py).
// The grouped R/K/V + low-rank task grid follows the design studied in
// BlinkDL/Albatross faster3b_2606, without its FP16 split-accumulator arithmetic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using at::Tensor;
__device__ __forceinline__ void copy16(void* dest,const void* src,bool valid) {
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dest));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(src),"r"(valid?16:0));
}
template<int CK> __device__ __forceinline__ int idx(int row,int col) {
    return row*CK+((col/8)^(row&(CK/8<8?CK/8-1:7)))*8;
}
template<class P> __device__ __forceinline__ void mma(float* d,const unsigned* a,const unsigned* b) {
    if constexpr(std::is_same_v<P,__half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
}
template<class P,int CK> __device__ __forceinline__ void load(P* s,const P* w,int k,int n,int tile,int step) {
    #pragma unroll
    for(int i=threadIdx.x;i<64*CK/8;i+=128) {
        int r=i/(CK/8),c=i%(CK/8)*8;
        copy16(s+idx<CK>(r,c),w+int64_t(tile*64+r)*k+step+c,tile*64+r<n);
    }
    asm volatile("cp.async.commit_group;":::"memory");
}
struct Ranks {
    const float* x[4];const void* w[4];int k[4],n,count;
};
template<class P> __device__ P narrow(float v) {
    if constexpr(std::is_same_v<P,__half>)return __float2half_rn(v);
    else return __float2bfloat16_rn(v);
}
template<class P> __global__ __launch_bounds__(128) void expand(Ranks p,float* out) {
    extern __shared__ __align__(16) unsigned char storage[];
    P* sb=(P*)storage;P* sx=sb+2*64*32;
    int g=blockIdx.y,k=p.k[g],tile=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int i=threadIdx.x;i<k;i+=128) {
        float v=p.x[g][i];
        if(g==0)v=tanhf(v);else if(g==2)v=1.f/(1.f+expf(-v));
        sx[i]=narrow<P>(v);
    }
    const P* w=(const P*)p.w[g];float acc[2][4]={{0.f}};
    load<P,32>(sb,w,k,p.n,tile,0);
    for(int block=0;block<k/32;++block) {
        __syncthreads();
        if(block+1<k/32) {
            load<P,32>(sb+((block+1)&1)*64*32,w,k,p.n,tile,(block+1)*32);
            asm volatile("cp.async.wait_group 1;":::"memory");
        } else asm volatile("cp.async.wait_group 0;":::"memory");
        __syncthreads();
        #pragma unroll
        for(int step=0;step<2;++step) {
            int offset=block*32+step*16+(lane%4)*2;
            unsigned a[4];a[0]=*(const unsigned*)(sx+offset);a[1]=a[0];
            a[2]=*(const unsigned*)(sx+offset+8);a[3]=a[2];
            unsigned b[4];
            unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(sb+(block&1)*64*32+idx<32>(warp*16+(lane/16)*8+lane%8,step*16+((lane/8)%2)*8)));
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(b[0]),"=r"(b[1]),"=r"(b[2]),"=r"(b[3]):"r"(addr));
            mma<P>(acc[0],a,b);mma<P>(acc[1],a,b+2);
        }
    }
    if(lane<4) {
        #pragma unroll
        for(int n=0;n<2;++n) {
            int col=tile*64+warp*16+n*8+lane*2;
            out[g*p.n+col]=acc[n][0];out[g*p.n+col+1]=acc[n][1];
        }
    }
}
Tensor run(std::vector<Tensor> xs,std::vector<Tensor> ws) {
    TORCH_CHECK((xs.size()==3 || xs.size()==4) && xs.size()==ws.size(),"groups");
    auto w=ws[0];TORCH_CHECK(w.is_cuda() && (w.scalar_type()==at::kHalf || w.scalar_type()==at::kBFloat16),"dtype");
    c10::cuda::CUDAGuard guard(w.device());Ranks p{};p.count=xs.size();p.n=w.size(0);
    TORCH_CHECK(p.n>0 && p.n%64==0,"output width");
    for(int g=0;g<p.count;++g) {
        TORCH_CHECK(xs[g].device()==w.device() && xs[g].scalar_type()==at::kFloat && xs[g].is_contiguous(),"FP32 input");
        TORCH_CHECK(ws[g].device()==w.device() && ws[g].scalar_type()==w.scalar_type() && ws[g].is_contiguous() && ws[g].dim()==2 && ws[g].size(0)==p.n,"weight");
        p.k[g]=ws[g].size(1);TORCH_CHECK(p.k[g]>0 && p.k[g]<=512 && p.k[g]%32==0 && xs[g].numel()==p.k[g],"rank");
        p.x[g]=xs[g].data_ptr<float>();p.w[g]=ws[g].data_ptr();
    }
    auto y=at::empty({p.count,p.n},w.options().dtype(at::kFloat));
    if(w.scalar_type()==at::kHalf)expand<__half><<<dim3(p.n/64,p.count),128,9216,c10::cuda::getCurrentCUDAStream()>>>(p,y.data_ptr<float>());
    else expand<__nv_bfloat16><<<dim3(p.n/64,p.count),128,9216,c10::cuda::getCurrentCUDAStream()>>>(p,y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

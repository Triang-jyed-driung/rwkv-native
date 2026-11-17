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
struct Groups {
    const void *x[7], *w[7];
    int n[7], splits[7], tiles[8], offsets[7], count, k;
};
__device__ __forceinline__ void copy16(void* dest,const void* src,bool valid) {
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dest));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(src),"r"(valid?16:0));
}
template<int CK> __device__ __forceinline__ int idx(int row,int col) {
    return row*CK+((col/8)^(row&7))*8;
}
template<class P> __device__ __forceinline__ void mma(float* d,const unsigned* a,const unsigned* b) {
    if constexpr(std::is_same_v<P,__half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};": "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
}
template<class P,int CK,int NT> __device__ __forceinline__ void load(P* s,const P* w,int k,int n,int tile,int step) {
    #pragma unroll
    for(int i=threadIdx.x;i<NT*CK/8;i+=128) {
        int r=i/(CK/8),c=i%(CK/8)*8;
        copy16(s+idx<CK>(r,c),w+int64_t(tile*NT+r)*k+step+c,tile*NT+r<n);
    }
    asm volatile("cp.async.commit_group;":::"memory");
}
template<class P,int CK,int NT> __device__ void body(Groups p,int g,int tile,P* smem,float* out) {
    const P *x=(const P*)p.x[g],*w=(const P*)p.w[g];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,segment=p.k/p.splits[g];
    float sum[2][4]={{0.f}};
    for(int split=0;split<p.splits[g];++split) {
        float acc[2][4]={{0.f}};
        load<P,CK,NT>(smem,w,p.k,p.n[g],tile,split*segment);
        for(int block=0;block<segment/CK;++block) {
            __syncthreads();
            if(block+1<segment/CK) {
                load<P,CK,NT>(smem+((block+1)&1)*NT*CK,w,p.k,p.n[g],tile,split*segment+(block+1)*CK);
                asm volatile("cp.async.wait_group 1;":::"memory");
            } else asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
            #pragma unroll
            for(int step=0;step<CK/16;++step) {
                if(warp<NT/16) {
                int offset=split*segment+block*CK+step*16+(lane%4)*2;
                unsigned a[4];a[0]=*(const unsigned*)(x+offset);a[1]=a[0];
                a[2]=*(const unsigned*)(x+offset+8);a[3]=a[2];
                unsigned b[4];
                unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(smem+(block&1)*NT*CK+idx<CK>(warp*16+(lane/16)*8+lane%8,step*16+((lane/8)%2)*8)));
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(b[0]),"=r"(b[1]),"=r"(b[2]),"=r"(b[3]):"r"(addr));
                mma<P>(acc[0],a,b);mma<P>(acc[1],a,b+2);
                }
            }
        }
        #pragma unroll
        for(int n=0;n<2;++n) {
            #pragma unroll
            for(int q=0;q<4;++q)sum[n][q]=split?__fadd_rn(sum[n][q],acc[n][q]):acc[n][q];
        }
        __syncthreads();
    }
    if(lane<4 && warp<NT/16) {
        #pragma unroll
        for(int n=0;n<2;++n) {
            int col=tile*NT+warp*16+n*8+lane*2;
            if(col<p.n[g]) {
                out[p.offsets[g]+col]=sum[n][0];out[p.offsets[g]+col+1]=sum[n][1];
            }
        }
    }
}
template<class P,int NT> __global__ __launch_bounds__(128) void grouped_decode(Groups p,float* out) {
    extern __shared__ __align__(16) unsigned char storage[];
    int g=0;while(g+1<p.count && blockIdx.x>=p.tiles[g+1])++g;
    int tile=blockIdx.x-p.tiles[g];
    if(p.splits[g]==1)body<P,256,NT>(p,g,tile,(P*)storage,out);
    else body<P,64,NT>(p,g,tile,(P*)storage,out);
}
Tensor run(std::vector<Tensor> xs,std::vector<Tensor> ws,std::vector<int64_t> splits,int64_t nt) {
    TORCH_CHECK(nt==16 || nt==32 || nt==64,"tile");
    TORCH_CHECK(!xs.empty() && xs.size()<=7 && xs.size()==ws.size() && xs.size()==splits.size(),"groups");
    auto x=xs[0];TORCH_CHECK(x.is_cuda() && (x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16),"dtype/device");
    c10::cuda::CUDAGuard guard(x.device());Groups p{};p.count=xs.size();p.k=x.size(1);int total=0;
    for(int g=0;g<p.count;++g) {
        TORCH_CHECK(xs[g].device()==x.device() && ws[g].device()==x.device() && xs[g].scalar_type()==x.scalar_type() && ws[g].scalar_type()==x.scalar_type(),"matching operands");
        TORCH_CHECK(xs[g].is_contiguous() && ws[g].is_contiguous() && xs[g].sizes()==at::IntArrayRef({1,p.k}) && ws[g].dim()==2 && ws[g].size(1)==p.k,"layout");
        p.n[g]=ws[g].size(0);p.splits[g]=splits[g];
        TORCH_CHECK(p.n[g]>0 && p.n[g]%8==0 && p.splits[g]>0 && p.k%p.splits[g]==0 && (p.k/p.splits[g])%(p.splits[g]==1?256:64)==0,"alignment");
        p.x[g]=xs[g].data_ptr();p.w[g]=ws[g].data_ptr();p.offsets[g]=total;total+=p.n[g];
        p.tiles[g+1]=p.tiles[g]+(p.n[g]+nt-1)/nt;
    }
    auto y=at::empty({total},x.options().dtype(at::kFloat));
    #define CALL(P,NT) {auto fn=grouped_decode<P,NT>;constexpr int bytes=2*NT*256*2;if(bytes>=49152)C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));fn<<<p.tiles[p.count],128,bytes,c10::cuda::getCurrentCUDAStream()>>>(p,y.data_ptr<float>());}
    #define LAUNCH(P) if(nt==16)CALL(P,16)else if(nt==32)CALL(P,32)else CALL(P,64)
    if(x.scalar_type()==at::kHalf){LAUNCH(__half)}else{LAUNCH(__nv_bfloat16)}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

// FP32 arithmetic copied from native wkv_prefetch.cuh (FlashRWKV2 MIT adaptation).
// Async group pipelining studied from Albatross; producer/consumer handshakes
// studied from /home/zrc/wkv7-kernels/wkv7_simt_prepend_operator/kernel.cu.
// Existing FlashRWKV2 attribution is retained in LICENSE.FlashRWKV2.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
template<int OFFSET> __device__ __forceinline__ void wkv_prefetch_reduce(float (&p)[16]) {
    #pragma unroll
    for(int i=0;i<OFFSET;++i)p[i]+=p[i+OFFSET];
    if constexpr(OFFSET>1)wkv_prefetch_reduce<OFFSET/2>(p);
}

template<int PREFETCH>
__device__ void fetch(float (&raw)[PREFETCH][6][64],const float*const(&src)[6],
                      int start,int T,int H,int batch,int head,int rank,int threads) {
    for(int z=rank;z<PREFETCH*6*16;z+=threads) {
        int token=z/(6*16),group=(z/16)%6,col=z%16*4;
        if(start+token<T) {
            long long off=((long long)batch*T+start+token)*H*64+head*64+col;
            unsigned dst=__cvta_generic_to_shared(&raw[token][group][col]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(src[group]+off):"memory");
        }
    }
    asm volatile("cp.async.commit_group;":::"memory");
}
template<int PREFETCH>
__device__ void decode(float (&raw)[PREFETCH][6][64],int rank,int threads,int valid_tokens) {
    for(int z=rank;z<valid_tokens*64;z+=threads) {
        float x=raw[z/64][1][z%64];
        raw[z/64][1][z%64]=expf(-0.6065306597126334f/(1.f+expf(-x)));
    }
}
template<int Base,bool Wait>
__device__ void handshake(int slot) {
    if constexpr(Wait) {
        if(slot==0)asm volatile("bar.sync %0,96;"::"n"(Base):"memory");
        else asm volatile("bar.sync %0,96;"::"n"(Base+2):"memory");
    } else {
        if(slot==0)asm volatile("bar.arrive %0,96;"::"n"(Base):"memory");
        else asm volatile("bar.arrive %0,96;"::"n"(Base+2):"memory");
    }
}
template<int PREFETCH,bool COOPERATIVE>
__global__ void pipelined(const float*r,const float*w,const float*k,const float*v,
                          const float*a,const float*b,const float*initial,const bool*mask,
                          float*y,float*final,float*cp,int T,int H,int chunk,int count) {
    constexpr bool SAVE=true;
    const int lane=threadIdx.x,batch=blockIdx.x/H,head=blockIdx.x%H;
    const long long base=(long long)blockIdx.x*64*64;
    const long long initial_batch_stride=(long long)H*64*64;
    __shared__ __align__(16) float raw[2][PREFETCH][6][64];
    const float*src[6]={r,w,k,v,a,b};
    if constexpr(COOPERATIVE) {
        if(lane>=64) {
            for(int start=0;start<T;start+=PREFETCH) {
                int slot=(start/PREFETCH)&1;
                if(start>=2*PREFETCH)handshake<2,true>(slot);
                fetch<PREFETCH>(raw[slot],src,start,T,H,batch,head,lane-64,32);
                asm volatile("cp.async.wait_group 0;":::"memory");
                __syncwarp();
                decode<PREFETCH>(raw[slot],lane-64,32,min(PREFETCH,T-start));
                __syncwarp();
                handshake<1,false>(slot);
            }
            return;
        }
    } else {
        fetch<PREFETCH>(raw[0],src,0,T,H,batch,head,lane,64);
    }
    float state[64];
    #pragma unroll
    for(int j=0;j<64;++j)state[j]=initial[(long long)batch*initial_batch_stride+(long long)head*64*64+j*64+lane];
    for(int start=0;start<T;start+=PREFETCH) {
        int slot=(start/PREFETCH)&1;
        if constexpr(COOPERATIVE)handshake<1,true>(slot);
        else {
            if(start+PREFETCH<T) {
                fetch<PREFETCH>(raw[slot^1],src,start+PREFETCH,T,H,batch,head,lane,64);
                asm volatile("cp.async.wait_group 1;":::"memory");
            } else asm volatile("cp.async.wait_group 0;":::"memory");
            __syncthreads();
            decode<PREFETCH>(raw[slot],lane,64,min(PREFETCH,T-start));
            __syncthreads();
        }
        auto &staged=raw[slot];
        #pragma unroll
        for(int within=0;within<PREFETCH;++within) {
            const int t=start+within;
            if(t<T) {
                if constexpr(SAVE) {
                    if(t%chunk==0) {
                        const int64_t cb=(int64_t(blockIdx.x)*count+t/chunk)*64*64;
                        #pragma unroll
                        for(int j=0;j<64;++j)cp[cb+j*64+lane]=state[j];
                    }
                }
                const int64_t index=((int64_t(batch)*T+t)*H+head)*64+lane;
                if(!mask[int64_t(batch)*T+t]) {y[index]=0.f;continue;}
                float part[16];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    float4 v0=*(const float4*)(&staged[within][4][j*4]);
                    float4 v1=*(const float4*)(&staged[within][4][j*4+16]);
                    float4 v2=*(const float4*)(&staged[within][4][j*4+32]);
                    float4 v3=*(const float4*)(&staged[within][4][j*4+48]);
                    part[j*4]=fmaf(state[j*4],v0.x,state[j*4+32]*v2.x)+fmaf(state[j*4+16],v1.x,state[j*4+48]*v3.x);
                    part[j*4+1]=fmaf(state[j*4+1],v0.y,state[j*4+33]*v2.y)+fmaf(state[j*4+17],v1.y,state[j*4+49]*v3.y);
                    part[j*4+2]=fmaf(state[j*4+2],v0.z,state[j*4+34]*v2.z)+fmaf(state[j*4+18],v1.z,state[j*4+50]*v3.z);
                    part[j*4+3]=fmaf(state[j*4+3],v0.w,state[j*4+35]*v2.w)+fmaf(state[j*4+19],v1.w,state[j*4+51]*v3.w);
                }
                wkv_prefetch_reduce<8>(part);
                const float dot=part[0],value=staged[within][3][lane];
                #pragma unroll
                for(int j=0;j<16;++j) {
                    float4 dv=*(const float4*)(&staged[within][1][j*4]);
                    float4 kv=*(const float4*)(&staged[within][2][j*4]);
                    float4 bv=*(const float4*)(&staged[within][5][j*4]);
                    state[j*4]=fmaf(state[j*4],dv.x,fmaf(dot,bv.x,kv.x*value));
                    state[j*4+1]=fmaf(state[j*4+1],dv.y,fmaf(dot,bv.y,kv.y*value));
                    state[j*4+2]=fmaf(state[j*4+2],dv.z,fmaf(dot,bv.z,kv.z*value));
                    state[j*4+3]=fmaf(state[j*4+3],dv.w,fmaf(dot,bv.w,kv.w*value));
                }
                #pragma unroll
                for(int j=0;j<4;++j) {
                    float4 v0=*(const float4*)(&staged[within][0][j*4]);
                    float4 v1=*(const float4*)(&staged[within][0][j*4+16]);
                    float4 v2=*(const float4*)(&staged[within][0][j*4+32]);
                    float4 v3=*(const float4*)(&staged[within][0][j*4+48]);
                    part[j*4]=fmaf(state[j*4],v0.x,state[j*4+32]*v2.x)+fmaf(state[j*4+16],v1.x,state[j*4+48]*v3.x);
                    part[j*4+1]=fmaf(state[j*4+1],v0.y,state[j*4+33]*v2.y)+fmaf(state[j*4+17],v1.y,state[j*4+49]*v3.y);
                    part[j*4+2]=fmaf(state[j*4+2],v0.z,state[j*4+34]*v2.z)+fmaf(state[j*4+18],v1.z,state[j*4+50]*v3.z);
                    part[j*4+3]=fmaf(state[j*4+3],v0.w,state[j*4+35]*v2.w)+fmaf(state[j*4+19],v1.w,state[j*4+51]*v3.w);
                }
                wkv_prefetch_reduce<8>(part);
                y[index]=part[0];
            }
        }

        if constexpr(COOPERATIVE) {
            if(start+2*PREFETCH<T)handshake<2,false>(slot);
        } else __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<64;++j)final[base+j*64+lane]=state[j];
}
std::vector<at::Tensor> run(std::vector<at::Tensor> xs,at::Tensor initial,
                           at::Tensor mask,int prefetch,bool cooperative) {
    TORCH_CHECK(xs.size()==6,"six inputs");
    auto x=xs[0];TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.dim()==4 && x.size(3)==64,"FP32 WKV inputs");
    c10::cuda::CUDAGuard guard(x.device());
    for(auto q:xs)TORCH_CHECK(q.sizes()==x.sizes() && q.is_contiguous() && q.scalar_type()==at::kFloat && q.device()==x.device(),"operands");
    int B=x.size(0),T=x.size(1),H=x.size(2),count=(T+15)/16;
    TORCH_CHECK(T>0 && B>0 && H>0,"nonempty input");
    TORCH_CHECK(initial.scalar_type()==at::kFloat && initial.device()==x.device() && initial.is_contiguous() && initial.sizes()==at::IntArrayRef({B,H,64,64}),"state");
    TORCH_CHECK(mask.scalar_type()==at::kBool && mask.device()==x.device() && mask.is_contiguous() && mask.sizes()==at::IntArrayRef({B,T}),"mask");
    auto y=at::empty_like(x),final=at::empty_like(initial),cp=at::empty({B,H,count,64,64},x.options());
    auto stream=c10::cuda::getCurrentCUDAStream();
    #define CALL(P,C) pipelined<P,C><<<B*H,C?96:64,0,stream>>>(xs[0].data_ptr<float>(),xs[1].data_ptr<float>(),xs[2].data_ptr<float>(),xs[3].data_ptr<float>(),xs[4].data_ptr<float>(),xs[5].data_ptr<float>(),initial.data_ptr<float>(),mask.data_ptr<bool>(),y.data_ptr<float>(),final.data_ptr<float>(),cp.data_ptr<float>(),T,H,16,count)
    #define SELECT(C) if(prefetch==4){CALL(4,C);}else if(prefetch==8){CALL(8,C);}else{CALL(16,C);}
    TORCH_CHECK(prefetch==4 || prefetch==8 || prefetch==16,"prefetch");
    if(cooperative){SELECT(true)}else{SELECT(false)}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y,final,cp};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

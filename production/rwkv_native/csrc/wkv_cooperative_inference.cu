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
        // Decay is precomputed by prepare_rows.
    }
}
template<int Base,bool Wait>
__device__ void handshake(int slot) {
    if constexpr(Wait) {
        if(slot==0)asm volatile("bar.sync %0,64;"::"n"(Base):"memory");
        else asm volatile("bar.sync %0,64;"::"n"(Base+2):"memory");
    } else {
        if(slot==0)asm volatile("bar.arrive %0,64;"::"n"(Base):"memory");
        else asm volatile("bar.arrive %0,64;"::"n"(Base+2):"memory");
    }
}
template<int PREFETCH,bool COOPERATIVE>
__global__ void pipelined(const float*r,const float*w,const float*k,const float*v,
                          const float*a,const float*b,const float*initial,const bool*mask,
                          float*y,float*final,float*cp,int T,int H,int chunk,int count) {
    constexpr bool SAVE=false;
    const int lane=threadIdx.x,bh=blockIdx.x/2,batch=bh/H,head=bh%H,value_offset=(blockIdx.x%2)*32;
    const long long base=(long long)bh*64*64+value_offset;
    const long long initial_batch_stride=(long long)H*64*64;
    __shared__ __align__(16) float raw[2][PREFETCH][6][64];
    const float*src[6]={r,w,k,v,a,b};
    if constexpr(COOPERATIVE) {
        if(lane>=32) {
            for(int start=0;start<T;start+=PREFETCH) {
                int slot=(start/PREFETCH)&1;
                if(start>=2*PREFETCH)handshake<2,true>(slot);
                fetch<PREFETCH>(raw[slot],src,start,T,H,batch,head,lane-32,32);
                asm volatile("cp.async.wait_group 0;":::"memory");
                __syncwarp();
                decode<PREFETCH>(raw[slot],lane-32,32,min(PREFETCH,T-start));
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
    for(int j=0;j<64;++j)state[j]=initial[(long long)batch*initial_batch_stride+(long long)head*64*64+j*64+value_offset+lane];
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
                const int64_t index=((int64_t(batch)*T+t)*H+head)*64+value_offset+lane;
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
                const float dot=part[0],value=staged[within][3][value_offset+lane];
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

using at::Tensor;
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for(int offset=16;offset;offset>>=1)x+=__shfl_down_sync(0xffffffff,x,offset);
    return __shfl_sync(0xffffffff,x,0);
}
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
std::vector<Tensor> run(std::vector<Tensor> xs,Tensor initial,Tensor mask,std::vector<Tensor> params,Tensor first,bool blend,int prefetch) {
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
#define WKV(P) pipelined<P,true><<<B*H*2,64,0,stream>>>(r.data_ptr<float>(),decay.data_ptr<float>(),key.data_ptr<float>(),value.data_ptr<float>(),negative.data_ptr<float>(),adapted.data_ptr<float>(),initial.data_ptr<float>(),mask.data_ptr<bool>(),y.data_ptr<float>(),final.data_ptr<float>(),nullptr,T,H,16,0)
 if(prefetch==4){WKV(4);}else{TORCH_CHECK(prefetch==8,"prefetch");WKV(8);}
#undef WKV
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,final,key,value};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

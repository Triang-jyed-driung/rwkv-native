// CUDA adaptation of FlashRWKV2 multi-token staging, with native FP32 arithmetic.
// See third_party/LICENSE.FlashRWKV2. No TileLang dependency.
#pragma once
__device__ __forceinline__ float fused_readout_sum(float x) {
    for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);
    return __shfl_sync(0xffffffff,x,0);
}
template<int OFFSET> __device__ __forceinline__ void wkv_prefetch_reduce(float (&p)[16]) {
    #pragma unroll
    for(int i=0;i<OFFSET;++i)p[i]+=p[i+OFFSET];
    if constexpr(OFFSET>1)wkv_prefetch_reduce<OFFSET/2>(p);
}
#include <torch/extension.h>
template<typename P,int PREFETCH,bool SAVE>
__global__ __launch_bounds__(64,8) void fused_wkv_prepare_readout_float(const float* __restrict__ r,const float* __restrict__ w,const float* __restrict__ k,const float* __restrict__ v,
                                   const float* __restrict__ a,const float* __restrict__ b,const float* __restrict__ initial,const bool* __restrict__ mask,
                                   P* __restrict__ y,float* __restrict__ final,float* __restrict__ cp,int T,int H,int chunk,int count,int64_t initial_batch_stride,const float* __restrict__ gate,const P* __restrict__ rk,const P* __restrict__ norm_weight,const P* __restrict__ norm_bias, const float* __restrict__ first_value, const P* __restrict__ w0, const P* __restrict__ k_scale, const P* __restrict__ a0, const P* __restrict__ k_adapt, const P* __restrict__ v0, bool blend) {
    const int lane=threadIdx.x,batch=blockIdx.x/H,head=blockIdx.x%H;
    const int64_t base=int64_t(blockIdx.x)*64*64;
    float state[64];
    __shared__ __align__(16) float staged[PREFETCH][7][64];
    __shared__ float recurrent[PREFETCH][64];
    #pragma unroll
    for(int j=0;j<64;++j)state[j]=initial[int64_t(batch)*initial_batch_stride+int64_t(head)*64*64+j*64+lane];
    for(int start=0;start<T;start+=PREFETCH) {
        for(int i=lane;i<PREFETCH*7*16;i+=64) {
            const int token=i/(7*16),group=(i/16)%7,offset=(i%16)*4;
            if(start+token<T) {
                const int64_t index=((int64_t(batch)*T+start+token)*H+head)*64+offset;
                unsigned target=static_cast<unsigned>(__cvta_generic_to_shared(&staged[token][group][offset]));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(target),"l"((group==0?r:group==1?w:group==2?k:group==3?v:group==4?a:group==5?b:gate)+index));
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        // Match key_forward_rows' two-values-per-lane reduction and parameter
        // expansion; consume raw gate/value vectors without global temporaries.
        for(int token=lane/32;token<PREFETCH;token+=2) {
            if(start+token<T) {
                float key[2],u[2],rate[2],sum=0.f;
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const int col=lane%32+j*32,c=head*64+col;
                    key[j]=staged[token][2][col];
                    u[j]=key[j]*float(k_scale[c]);
                    rate[j]=1.f/(1.f+expf(-(staged[token][4][col]+float(a0[c]))));
                    sum+=u[j]*u[j];
                }
                const float inverse=1.f/fmaxf(sqrtf(fused_readout_sum(sum)),1e-12f);
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const int col=lane%32+j*32,c=head*64+col;
                    const int64_t index=((int64_t(batch)*T+start+token)*H+head)*64+col;
                    const float q=u[j]*inverse;
                    float value=staged[token][3][col];
                    if(blend) {
                        const float vg=1.f/(1.f+expf(-__fadd_rn(staged[token][5][col],float(v0[c]))));
                        value=__fadd_rn(value,__fmul_rn(first_value[index]-value,vg));
                    }
                    const float decay=__fadd_rn(staged[token][1][col],float(w0[c]));
                    staged[token][1][col]=expf(-0.6065306597126334f/(1.f+expf(-decay)));
                    staged[token][2][col]=key[j]*(1.f+(rate[j]-1.f)*float(k_adapt[c]));
                    staged[token][3][col]=value;
                    staged[token][4][col]=-q;
                    staged[token][5][col]=q*rate[j];
                }
            }
        }
        __syncthreads();
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
                if(!mask[int64_t(batch)*T+t]) {recurrent[within][lane]=0.f;continue;}
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
                recurrent[within][lane]=part[0];
            }
        }
        __syncthreads();
        // One warp normalizes a full 64-value head. Stage all token outputs
        // before readout, preserving the original two-values-per-lane tree.
        const int warp=lane/32, column_lane=lane%32;
        for(int within=warp;within<PREFETCH;within+=2) {
            const int t=start+within;
            if(t<T) {
                float x[2],dot=0.f,sum=0.f;
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const int col=column_lane+j*32,c=head*64+col;
                    x[j]=recurrent[within][col];
                    sum+=x[j];
                    dot+=staged[within][0][col]*staged[within][2][col]*float(rk[c]);
                }
                const float mean=fused_readout_sum(sum)/64.f;
                sum=0.f;
                #pragma unroll
                for(int j=0;j<2;++j) {float d=x[j]-mean;sum+=d*d;}
                const float inv=rsqrtf(fused_readout_sum(sum)/64.f+64e-5f);
                dot=fused_readout_sum(dot);
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const int col=column_lane+j*32,c=head*64+col;
                    const int64_t index=((int64_t(batch)*T+t)*H+head)*64+col;
                    y[index]=P(((x[j]-mean)*inv*float(norm_weight[c])+float(norm_bias[c])+
                                dot*staged[within][3][col])*staged[within][6][col]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<64;++j)final[base+j*64+lane]=state[j];
}

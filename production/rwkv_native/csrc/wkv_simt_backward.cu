// FP32 adaptation of wkv7_simt_prepend_operator reverse_apply from /home/zrc/wkv7-kernels.
// Native checkpoint replay supplies S*a; scalar inverse reconstructs at most 16 steps.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
// Fixed N64 BF16 / FP32 WKV7 training operator for sm89.
// Forward: fixed 8192-token affine chunks; backward: fixed 4096-token chunks.
// Both boundary builders use scalar FP32 composition, without matrix inverses.
// Only an exactly-zero FP32 map permits early exit; pending copies are drained.
// Checkpoint interval 16, with the same bijective R64C4 layout in both directions.
// Exact parallel reverse affine scan, with ordinary FP32 scalar GEMM prefix.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
using scalar=float;
constexpr int _CHUNK_LEN_=16;
__device__ float as_float(scalar x) { return x; }
__device__ scalar as_scalar(float x) { return x; }
__device__ float decay(float x,float& factor) {
 float sig=1.f/(1.f+expf(-x));
 float w=expf(-0.6065306597f*sig);
 factor=-0.6065306597f*w*sig*(1.f-sig); return w;
}
__device__ __forceinline__ int checkpoint_index(int key,int value) {
 return ((key/64)*16+value/4)*(64*4)+(key%64)*4+value%4;
}
// Inspired by Albatross's vector cp.async staging. State/arithmetic stay FP32.
// Double-buffer raw inputs, prefetch the next tile during the current recurrence.
__device__ __forceinline__ void copy_async16(void* dst,const void* src,bool valid=true) {
    unsigned address=__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(src),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ void commit_async() {
    asm volatile("cp.async.commit_group;":::"memory");
}
__device__ __forceinline__ void wait_async() {
    asm volatile("cp.async.wait_group 0;":::"memory");
}

template<int N,int Tile,int Count>
__device__ __forceinline__ void prefetch_inputs(scalar (&raw)[Count][Tile][N],
    const scalar* const (&src)[Count],int64_t base,int t0,int H) {
    constexpr int Elements=16/sizeof(scalar),Vectors=N/Elements;
#pragma unroll
    for(int q=0;q<Count;++q) {
#pragma unroll
        for(int z=0;z<(Tile*Vectors+N-1)/N;++z) {
            int flat=threadIdx.x+z*N;
            if(flat<Tile*Vectors) {
                int tt=flat/Vectors,j=(flat%Vectors)*Elements;
                copy_async16(&raw[q][tt][j],src[q]+base+int64_t(t0+tt)*H*N+j);
            }
        }
    }
}

template<int N,int Tile>
__device__ __forceinline__ void prefetch_sa(float (&raw)[Tile][N],
    const float* src,int64_t base,int t0,int H) {
#pragma unroll
    for(int z=0;z<(Tile*(N/4)+N-1)/N;++z) {
        int flat=threadIdx.x+z*N;
        if(flat<Tile*(N/4)) {
            int tt=flat/(N/4),j=(flat%(N/4))*4;
            copy_async16(&raw[tt][j],src+base+int64_t(t0+tt)*H*N+j);
        }
    }
}

#ifndef BLOCK_T
#define BLOCK_T 4096
#endif
template<int Tile>
__device__ __forceinline__ void build_prefetch(scalar(&raw)[5][Tile][64],const scalar*const(&src)[5],int64_t base,int t0,int H,int end){
 constexpr int Size=Tile*16,Groups=64/Size;
 int group=threadIdx.x/Size,flat=threadIdx.x%Size;
#pragma unroll
 for(int q=0;q<5;q++)if(q%Groups==group)copy_async16(&raw[q][flat/16][flat%16*4],src[q]+base+int64_t(t0+flat/16)*H*64+flat%16*4,t0+flat/16<end);
}
constexpr int RT=4;
// Compose each affine map by prepending future steps. One thread owns a
// column of M and C. Once every FP32 entry of M is exactly zero, all later
// contributions to C are exactly zero for finite inputs; no threshold is used.
__global__ void reverse_build(int T,int H,const scalar*r,const scalar*w,
 const scalar*a,const scalar*b,const scalar*dy,const bool*mask,float*M,float*C){
 int j=threadIdx.x,chunk=blockIdx.z;
 if(chunk+1==(T+BLOCK_T-1)/BLOCK_T)return;
 int begin=(((T+BLOCK_T-1)/BLOCK_T)-1-chunk)*BLOCK_T,end=min(T,begin+BLOCK_T);
 int64_t base=(int64_t(blockIdx.y)*T*H+blockIdx.x)*64;
 const scalar*src[5]={r,w,a,b,dy};
 __shared__ __align__(16) scalar raw[3][5][RT][64];
 __shared__ float decoded[RT][5][64];
 float m[64]={},c[64]={};
#pragma unroll
 for(int i=0;i<64;i++)m[i]=(i==j)?1.f:0.f;
 if(begin+0*RT<end){build_prefetch<RT>(raw[0],src,base,begin+0*RT,H,end);commit_async();}
if(begin+1*RT<end){build_prefetch<RT>(raw[1],src,base,begin+1*RT,H,end);commit_async();}
 for(int t0=begin;t0<end;t0+=RT){
  int stage=((t0-begin)/RT)%3;if(t0+2*RT<end)asm volatile("cp.async.wait_group 1;":::"memory");else wait_async();__syncthreads();
  if(t0+2*RT<end){build_prefetch<RT>(raw[(stage+2)%3],src,base,t0+2*RT,H,end);commit_async();}
#pragma unroll 1
  for(int t=0;t<RT;t++){
#pragma unroll
   for(int q=0;q<5;q++)decoded[t][q][j]=as_float(raw[stage][q][t][j]);
   float f;decoded[t][1][j]=decay(decoded[t][1][j],f);
  }
  __syncthreads();
#pragma unroll 
  for(int t=0;t<RT;t++){
   if(t0+t>=end || !mask[int64_t(blockIdx.y)*T+t0+t])continue;
   auto &x=decoded[t];
   float ap[4]={},rp[4]={};
#pragma unroll
   for(int i=0;i<64;i++)ap[i%4]+=x[2][i]*m[i];
   float am=ap[0]+ap[1]+ap[2]+ap[3];
#pragma unroll
   for(int i=0;i<64;i++){
    m[i]=x[1][i]*m[i]+x[3][i]*am;
    rp[i%4]+=x[0][i]*m[i];
   }
   float rm=rp[0]+rp[1]+rp[2]+rp[3];
#pragma unroll
   for(int i=0;i<64;i++)c[i]+=x[4][i]*rm;
  }
  if(((t0-begin)/RT)%8==7){
   bool any=false;
#pragma unroll
   for(int i=0;i<64;i++)any|=(__float_as_uint(m[i])&0x7fffffffU)!=0;
   if(!__syncthreads_or(any)){wait_async();__syncthreads();break;}
  }
 }
 int64_t off=((int64_t(blockIdx.y)*H+blockIdx.x)*((T+BLOCK_T-1)/BLOCK_T)+chunk)*4096;
#pragma unroll
 for(int i=0;i<64;i++){M[off+j*64+i]=m[i];C[off+j*64+i]=c[i];}
}
__device__ int left_index(int row,int key){return key^(((row/4)&7)*4);}
__device__ int right_index(int row,int key){
 return ((key&3)|((key&4)<<3)|(key&24)|((key&32)>>3))^((row&7)*4);
}
__global__ void affine_prefix(int T,int H,const float*M,const float*C,const float*terminal,float*initial){
 int tid=threadIdx.x,m=tid/8*4,n=tid%8*8;
 int64_t base=(int64_t(blockIdx.y)*H+blockIdx.x)*((T+BLOCK_T-1)/BLOCK_T)*4096;
 __shared__ __align__(16) float left[64][64],right[64][64];
 for(int flat=tid;flat<4096;flat+=128){
  int value=flat/64,key=flat%64;
  left[value][left_index(value,key)]=terminal?terminal[(int64_t(blockIdx.y)*H+blockIdx.x)*4096+key*64+value]:0.f;
 }
 __syncthreads();
 for(int chunk=0;chunk<((T+BLOCK_T-1)/BLOCK_T);chunk++){
  int64_t off=base+int64_t(chunk)*4096;
  for(int flat=tid;flat<4096;flat+=128){
   int value=flat%64,key=flat/64;
   initial[off+flat]=left[value][left_index(value,key)];
  }
  if(chunk+1==((T+BLOCK_T-1)/BLOCK_T))break;
  for(int flat=tid;flat<4096;flat+=128){
   int row=flat%64,key=flat/64;
   right[row][right_index(row,key)]=M[off+flat];
  }
  __syncthreads();
  float out[4][8]={};
#pragma unroll 1
  for(int d=0;d<64;d+=4){
   float4 av[4],lo[4],hi[4];
#pragma unroll
   for(int j=0;j<4;j++){
    av[j]=*reinterpret_cast<float4*>(&left[m+j][left_index(m+j,d)]);
    lo[j]=*reinterpret_cast<float4*>(&right[d+j][right_index(d+j,n)]);
    hi[j]=*reinterpret_cast<float4*>(&right[d+j][right_index(d+j,n+4)]);
   }
#pragma unroll
   for(int i=0;i<4;i++){
    float a[4]={av[i].x,av[i].y,av[i].z,av[i].w};
#pragma unroll
    for(int j=0;j<4;j++){
     out[i][0]+=a[j]*lo[j].x;out[i][1]+=a[j]*lo[j].y;out[i][2]+=a[j]*lo[j].z;out[i][3]+=a[j]*lo[j].w;
     out[i][4]+=a[j]*hi[j].x;out[i][5]+=a[j]*hi[j].y;out[i][6]+=a[j]*hi[j].z;out[i][7]+=a[j]*hi[j].w;
    }
   }
  }
  // Retire both matrix readers before replacing the state and next right tile.
  __syncthreads();
#pragma unroll
  for(int j=0;j<8;j++){
   float4 c=*reinterpret_cast<const float4*>(C+off+(n+j)*64+m);
   out[0][j]+=c.x;out[1][j]+=c.y;out[2][j]+=c.z;out[3][j]+=c.w;
  }
#pragma unroll
  for(int i=0;i<4;i++){
   *reinterpret_cast<float4*>(&left[m+i][left_index(m+i,n)])=make_float4(out[i][0],out[i][1],out[i][2],out[i][3]);
   *reinterpret_cast<float4*>(&left[m+i][left_index(m+i,n+4)])=make_float4(out[i][4],out[i][5],out[i][6],out[i][7]);
  }
  __syncthreads();
 }
}

using at::Tensor;
constexpr float SCALE=-0.6065306597126334f;
__device__ float decay(float w){return expf(SCALE/(1.f+expf(-w)));}
__global__ void prepare_sa(const float*w,const float*k,const float*v,const float*a,const float*b,
 const float*cp,const bool*mask,float*sa,float*last,int T,int H,int count){
 int value=threadIdx.x,batch=blockIdx.x/H,head=blockIdx.x%H,c=blockIdx.y;
 int start=c*16,stop=min(T,start+16);float state[64];
 __shared__ float x[5][64];
 int64_t cb=(int64_t(blockIdx.x)*count+c)*4096;
 #pragma unroll
 for(int j=0;j<64;j++)state[j]=cp[cb+j*64+value];
 for(int t=start;t<stop;t++){
  int64_t off=((int64_t(batch)*T+t)*H+head)*64;
  if(!mask[int64_t(batch)*T+t]){sa[off+value]=0.f;continue;}
  __syncthreads();x[0][value]=decay(w[off+value]);x[1][value]=k[off+value];x[2][value]=v[off+value];x[3][value]=a[off+value];x[4][value]=b[off+value];__syncthreads();
  float part[16];
  #pragma unroll
  for(int j=0;j<16;j++)part[j]=fmaf(state[j],x[3][j],state[j+32]*x[3][j+32])+fmaf(state[j+16],x[3][j+16],state[j+48]*x[3][j+48]);
  #pragma unroll
  for(int step=8;step;step/=2){
   #pragma unroll
   for(int j=0;j<step;j++)part[j]+=part[j+step];
  }
  float dot=part[0],vv=x[2][value];sa[off+value]=dot;
  #pragma unroll
  for(int j=0;j<64;j++)state[j]=fmaf(state[j],x[0][j],fmaf(dot,x[4][j],x[1][j]*vv));
 }
 if(c==count-1){
  #pragma unroll
  for(int j=0;j<64;j++)last[int64_t(blockIdx.x)*4096+j*64+value]=state[j];
 }
}
__global__ void reverse_apply(const float*r,const float*w,const float*k,const float*v,const float*a,const float*b,
 const float*dy,const float*cp,const float*sa,const float*last,const float*terminal,const bool*mask,
 float*dr,float*dw,float*dk,float*dv,float*da,float*db,float*ds,int T,int H,int count,int segments){
 int i=threadIdx.x,batch=blockIdx.x/H,head=blockIdx.x%H;
 int64_t base=int64_t(blockIdx.x)*4096;
 int chunk=blockIdx.y,begin=segments==1?0:(segments-1-chunk)*BLOCK_T,end=segments==1?T:min(T,begin+BLOCK_T);
 int64_t adjoint_base=(int64_t(blockIdx.x)*segments+chunk)*4096;
 float state[64],drow[64],dcol[64];
 __shared__ float x[9][64],dsb[64];
 #pragma unroll
 for(int j=0;j<64;j++){drow[j]=terminal?terminal[adjoint_base+j*64+i]:0.f;dcol[j]=terminal?terminal[adjoint_base+i*64+j]:0.f;}
 for(int t=end-1;t>=begin;t--){
  int64_t off=((int64_t(batch)*T+t)*H+head)*64,idx=off+i;
  if(t==T-1 || (t+1)%16==0){
   const float*end=(t==T-1)?last+base:cp+(int64_t(blockIdx.x)*count+(t+1)/16)*4096;
   #pragma unroll
   for(int j=0;j<64;j++)state[j]=end[i*64+j];
  }
  if(!mask[int64_t(batch)*T+t]){dr[idx]=dw[idx]=dk[idx]=dv[idx]=da[idx]=db[idx]=0.f;continue;}
  __syncthreads();
  x[0][i]=r[idx];float sig=1.f/(1.f+expf(-w[idx]));float wi=expf(SCALE*sig);
  x[1][i]=wi;x[2][i]=k[idx];x[3][i]=v[idx];x[4][i]=a[idx];x[5][i]=b[idx];x[6][i]=dy[idx];x[7][i]=sa[idx];x[8][i]=SCALE*wi*sig*(1.f-sig);__syncthreads();
  float drp[2]={};
  #pragma unroll
  for(int j=0;j<64;j++)drp[j%2]+=state[j]*x[6][j];
  dr[idx]=drp[0]+drp[1];
  float ri=x[0][i],ki=x[2][i],ai=x[4][i],bi=x[5][i],dyi=x[6][i],iwi=1.f/wi;
  #pragma unroll
  for(int j=0;j<64;j++){
   state[j]=(state[j]-ki*x[3][j]-bi*x[7][j])*iwi;
   drow[j]+=dyi*x[0][j];dcol[j]+=ri*x[6][j];
  }
  float wp[2]={},kp[2]={},bp[2]={},vp[2]={},sp[2]={};
  #pragma unroll
  for(int j=0;j<64;j++){
   wp[j%2]+=dcol[j]*state[j];kp[j%2]+=dcol[j]*x[3][j];bp[j%2]+=dcol[j]*x[7][j];
   vp[j%2]+=drow[j]*x[2][j];sp[j%2]+=drow[j]*x[5][j];
  }
  dw[idx]=(wp[0]+wp[1])*x[8][i];dk[idx]=kp[0]+kp[1];db[idx]=bp[0]+bp[1];dv[idx]=vp[0]+vp[1];
  float dsa=sp[0]+sp[1];dsb[i]=dsa;__syncthreads();
  float ap[2]={};
  #pragma unroll
  for(int j=0;j<64;j++)ap[j%2]+=state[j]*dsb[j];
  da[idx]=ap[0]+ap[1];
  #pragma unroll
  for(int j=0;j<64;j++){drow[j]=drow[j]*x[1][j]+dsa*x[4][j];dcol[j]=dcol[j]*wi+ai*dsb[j];}
 }
 #pragma unroll
 for(int j=0;j<64;j++)if(chunk==segments-1)ds[base+j*64+i]=drow[j];
}
std::vector<Tensor> run(std::vector<Tensor> xs,Tensor mask,Tensor cp,Tensor dy,Tensor terminal,int ignored){
 TORCH_CHECK(xs.size()==6,"six inputs");auto x=xs[0];
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.dim()==4 && x.size(3)==64,"FP32 inputs");
 c10::cuda::CUDAGuard guard(x.device());
 int B=x.size(0),T=x.size(1),H=x.size(2),count=(T+15)/16;
 TORCH_CHECK(B>0 && T>0 && H>0,"nonempty");
 for(auto q:xs)TORCH_CHECK(q.sizes()==x.sizes() && q.device()==x.device() && q.scalar_type()==at::kFloat && q.is_contiguous(),"operands");
 TORCH_CHECK(dy.sizes()==x.sizes() && dy.device()==x.device() && dy.scalar_type()==at::kFloat && dy.is_contiguous(),"dy");
 TORCH_CHECK(mask.sizes()==at::IntArrayRef({B,T}) && mask.device()==x.device() && mask.scalar_type()==at::kBool && mask.is_contiguous(),"mask");
 TORCH_CHECK(cp.sizes()==at::IntArrayRef({B,H,count,64,64}) && cp.device()==x.device() && cp.scalar_type()==at::kFloat && cp.is_contiguous(),"cp");
 TORCH_CHECK(terminal.device()==x.device() && terminal.scalar_type()==at::kFloat && terminal.is_contiguous() && (terminal.numel()==0 || terminal.sizes()==at::IntArrayRef({B,H,64,64})),"terminal");
 auto sa=at::empty_like(x),last=at::empty({B,H,64,64},x.options()),ds=at::empty_like(last);
 std::vector<Tensor> grads;for(int i=0;i<6;i++)grads.push_back(at::empty_like(x));
 auto stream=c10::cuda::getCurrentCUDAStream();
 prepare_sa<<<dim3(B*H,count),64,0,stream>>>(xs[1].data_ptr<float>(),xs[2].data_ptr<float>(),xs[3].data_ptr<float>(),xs[4].data_ptr<float>(),xs[5].data_ptr<float>(),cp.data_ptr<float>(),mask.data_ptr<bool>(),sa.data_ptr<float>(),last.data_ptr<float>(),T,H,count);
 int segments=(B*H<128)?(T+BLOCK_T-1)/BLOCK_T:1;
 Tensor workspace;const float*incoming=terminal.numel()?terminal.data_ptr<float>():nullptr;
 if(segments>1){
  workspace=at::empty({3,B,H,segments,64,64},x.options());
  int64_t n=int64_t(B)*H*segments*4096;float*M=workspace.data_ptr<float>(),*C=M+n,*initial=C+n;
  reverse_build<<<dim3(H,B,segments),64,0,stream>>>(T,H,xs[0].data_ptr<float>(),xs[1].data_ptr<float>(),xs[4].data_ptr<float>(),xs[5].data_ptr<float>(),dy.data_ptr<float>(),mask.data_ptr<bool>(),M,C);
  affine_prefix<<<dim3(H,B),128,0,stream>>>(T,H,M,C,incoming,initial);incoming=initial;
 }
 reverse_apply<<<dim3(B*H,segments),64,0,stream>>>(xs[0].data_ptr<float>(),xs[1].data_ptr<float>(),xs[2].data_ptr<float>(),xs[3].data_ptr<float>(),xs[4].data_ptr<float>(),xs[5].data_ptr<float>(),dy.data_ptr<float>(),cp.data_ptr<float>(),sa.data_ptr<float>(),last.data_ptr<float>(),incoming,mask.data_ptr<bool>(),grads[0].data_ptr<float>(),grads[1].data_ptr<float>(),grads[2].data_ptr<float>(),grads[3].data_ptr<float>(),grads[4].data_ptr<float>(),grads[5].data_ptr<float>(),ds.data_ptr<float>(),T,H,count,segments);
 C10_CUDA_KERNEL_LAUNCH_CHECK();grads.push_back(ds);return grads;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

// SPDX-License-Identifier: MIT
// Static SM80 MMA tile; grouped role adaptation. See PRE_GROUP_PROVENANCE.json.
__global__ __launch_bounds__(128,1) void pre_partial_m16(PreGroups p,float* out,int64_t rows){

 int task=blockIdx.x,g=0;
 while(g+1<p.count && task>=((p.n[g]+63)/64)*p.splits[g]){
  task-=((p.n[g]+63)/64)*p.splits[g];++g;
 }
 const int tiles=(p.n[g]+63)/64;
 const int local_bn=task%tiles;
 const int inner_blocks=64/p.splits[g];
 const int64_t k_blocks=64,out_blocks=p.n[g]/16;

 const int part=task/tiles;
 const half_t* X=p.x[g]+part*(2048/p.splits[g]);
 const half_t* W=p.w[g]+part*(2048/p.splits[g]);
 float* Y=out+int64_t(part)*p.total_n*rows+int64_t(p.prefix[g])*rows;
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* b = ((void*)((char*)buf_dyn_shmem + 0));
  void* a = ((void*)((char*)buf_dyn_shmem + 8192));
  float c[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(c + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (0 < inner_blocks) {
    decode::cp_async_gs_conditional<8>((&(((half_t*)a)[(((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(X[(((((((int64_t)blockIdx.y) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) * k_blocks) * (int64_t)32) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4))])), ((((((int64_t)blockIdx.y) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) < rows) && (((((int64_t)blockIdx.y) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) < rows)));
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      decode::cp_async_gs_conditional<16>((&(((half_t*)b)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(W[((((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) * k_blocks) * (int64_t)32) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8))])), (((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16)) && ((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16))));
    }
    decode::cp_async_commit();
  }
  for (int k = 0; k < (inner_blocks - 1); ++k) {
    __syncthreads();
    decode::cp_async_gs_conditional<8>((&(((half_t*)a)[(((((((k + 1) & 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(X[((((((int64_t)k) * (int64_t)32) + ((((((int64_t)blockIdx.y) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) * k_blocks) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)4)) + (int64_t)32)])), ((((((int64_t)blockIdx.y) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) < rows) && (((((int64_t)blockIdx.y) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) < rows)));
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      decode::cp_async_gs_conditional<16>((&(((half_t*)b)[(((((((k + 1) & 1) * 2048) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(W[((((((int64_t)k) * (int64_t)32) + (((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_2) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) * k_blocks) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32)])), (((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_2) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16)) && ((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_2) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16))));
    }
    decode::cp_async_commit();
    decode::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[8];
      for (int ki = 0; ki < 2; ++ki) {
        decode::ptx_ldmatrix_x4((&(((half_t*)a)[(((((k & 1) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[0])));
        decode::ptx_ldmatrix_x4((&(((half_t*)b)[((((((k & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[0])));
        decode_mma(reinterpret_cast<float*>(c + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        decode_mma(reinterpret_cast<float*>(c + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
  }
  decode::cp_async_wait<0>();
  __syncthreads();
  if (1 <= inner_blocks) {
    {
      half_t A_local_1[8];
      half_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        decode::ptx_ldmatrix_x4((&(((half_t*)a)[((((((inner_blocks + 1) & 1) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
        decode::ptx_ldmatrix_x4((&(((half_t*)b)[(((((((inner_blocks + 1) & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_1[0])));
        decode_mma(reinterpret_cast<float*>(c + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        decode_mma(reinterpret_cast<float*>(c + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    float Y_local_cast[2];
    if (((((((int64_t)blockIdx.y) * (int64_t)16) + ((((int64_t)i_3) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) < rows) && ((((((int64_t)local_bn) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (int64_t)5)) - out_blocks) < (int64_t)0)) {
      *(float2*)(Y_local_cast + 0) = *(float2*)(c + (i_3 * 2));
      *(float2*)(Y + (((((((int64_t)local_bn) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((((int64_t)blockIdx.y) * (int64_t)16) + ((((int64_t)i_3) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * out_blocks) * (int64_t)16)) + ((((int64_t)i_3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(Y_local_cast + 0);
    }
  }
}

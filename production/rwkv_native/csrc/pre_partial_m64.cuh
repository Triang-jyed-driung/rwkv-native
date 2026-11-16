// SPDX-License-Identifier: MIT
// Static SM80 MMA tile; grouped role adaptation. See PRE_GROUP_PROVENANCE.json.
__global__ __launch_bounds__(128,1) void pre_partial_m64(PreGroups p,float* out,int64_t rows){

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
  void* a = ((void*)((char*)buf_dyn_shmem + 0));
  void* b = ((void*)((char*)buf_dyn_shmem + 8192));
  float c[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(c + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  if (0 < inner_blocks) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      decode::cp_async_gs_conditional<16>((&(((half_t*)a)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(X[((((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) * k_blocks) * (int64_t)32) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8))])), (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows) && ((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      decode::cp_async_gs_conditional<16>((&(((half_t*)b)[((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(W[((((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_2) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) * k_blocks) * (int64_t)32) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8))])), (((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_2) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16)) && ((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_2) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16))));
    }
    decode::cp_async_commit();
  }
  for (int k = 0; k < (inner_blocks - 1); ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      decode::cp_async_gs_conditional<16>((&(((half_t*)a)[(((((((k + 1) & 1) * 2048) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(X[((((((int64_t)k) * (int64_t)32) + (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_3) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) * k_blocks) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32)])), (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_3) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows) && ((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_3) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      decode::cp_async_gs_conditional<16>((&(((half_t*)b)[(((((((k + 1) & 1) * 2048) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(W[((((((int64_t)k) * (int64_t)32) + (((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_4) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) * k_blocks) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32)])), (((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_4) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16)) && ((((((int64_t)local_bn) * (int64_t)64) + (((int64_t)i_4) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < (out_blocks * (int64_t)16))));
    }
    decode::cp_async_commit();
    decode::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local[16];
      half_t B_local[16];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          decode::ptx_ldmatrix_x4((&(((half_t*)a)[(((((((k & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_5 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_5 * 8)])));
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          decode::ptx_ldmatrix_x4((&(((half_t*)b)[((((((((k & 1) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (i_6 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[(i_6 * 8)])));
        }
        for (int i_7 = 0; i_7 < 2; ++i_7) {
          for (int j = 0; j < 2; ++j) {
            decode_mma(reinterpret_cast<float*>(c + ((i_7 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_7 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            decode_mma(reinterpret_cast<float*>(c + (((i_7 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  decode::cp_async_wait<0>();
  __syncthreads();
  if (1 <= inner_blocks) {
    {
      half_t A_local_1[16];
      half_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          decode::ptx_ldmatrix_x4((&(((half_t*)a)[((((((((inner_blocks + 1) & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_8 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[(i_8 * 8)])));
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          decode::ptx_ldmatrix_x4((&(((half_t*)b)[(((((((((inner_blocks + 1) & 1) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (i_9 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_1[(i_9 * 8)])));
        }
        for (int i_10 = 0; i_10 < 2; ++i_10) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            decode_mma(reinterpret_cast<float*>(c + ((i_10 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            decode_mma(reinterpret_cast<float*>(c + (((i_10 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 16; ++i_11) {
    float Y_local_cast[2];
    if (((((((((int64_t)blockIdx.y) * (int64_t)64) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_11) >> (int64_t)3) * (int64_t)16)) + ((((int64_t)i_11) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) < rows) && (((((((int64_t)local_bn) * (int64_t)4) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)2)) + ((((int64_t)i_11) & (int64_t)7) >> (int64_t)2)) - out_blocks) < (int64_t)0)) {
      *(float2*)(Y_local_cast + 0) = *(float2*)(c + (i_11 * 2));
      *(float2*)(Y + (((((((int64_t)local_bn) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32)) + (((((((((int64_t)blockIdx.y) * (int64_t)64) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_11) >> (int64_t)3) * (int64_t)16)) + ((((int64_t)i_11) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * out_blocks) * (int64_t)16)) + (((((int64_t)i_11) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(Y_local_cast + 0);
    }
  }
}

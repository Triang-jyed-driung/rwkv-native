// SPDX-License-Identifier: MIT
// Adapted FlashRWKV2 grouped projection; see third_party/RANK_GROUP_PROVENANCE.json.
#include "rank_group_ptx.cuh"
extern "C" __global__ void rank_group_4_m64(const half_t* __restrict__ W0, const half_t* __restrict__ W1, const half_t* __restrict__ W2, const half_t* __restrict__ W3, const half_t* __restrict__ X0, const half_t* __restrict__ X1, const half_t* __restrict__ X2, const half_t* __restrict__ X3, float* __restrict__ Y, int64_t rows);
extern "C" __global__ void __launch_bounds__(128, 1) rank_group_4_m64(const half_t* __restrict__ W0, const half_t* __restrict__ W1, const half_t* __restrict__ W2, const half_t* __restrict__ W3, const half_t* __restrict__ X0, const half_t* __restrict__ X1, const half_t* __restrict__ X2, const half_t* __restrict__ X3, float* __restrict__ Y, int64_t rows) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* ws = ((void*)((char*)buf_dyn_shmem + 0));
  void* xs = ((void*)((char*)buf_dyn_shmem + 8192));
  float acc[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  int condval;
  if ((((int)blockIdx.x) < 32)) {
    condval = 0;
  } else {
    int condval_1;
    if ((((int)blockIdx.x) < 64)) {
      condval_1 = 1;
    } else {
      int condval_2;
      if ((((int)blockIdx.x) < 96)) {
        condval_2 = 2;
      } else {
        condval_2 = 3;
      }
      condval_1 = condval_2;
    }
    condval = condval_1;
  }
  if (condval == 0) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      half_t X0_local_cast_1[8];
      half_t xs_local_cast[8];
      half_t broadcast_var_1 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
      uint4 condval_3;
      if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
        condval_3 = *(uint4*)(X0 + ((((((int64_t)blockIdx.y) * (int64_t)6144) + (((int64_t)i_1) * (int64_t)3072)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)96)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)));
      } else {
        condval_3 = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(X0_local_cast_1 + 0) = condval_3;
      for (int vec = 0; vec < 4; ++vec) {
        half_t broadcast_var_2 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
        uint1 condval_4;
        if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
          condval_4 = *(uint1*)(X0_local_cast_1 + (vec * 2));
        } else {
          condval_4 = make_uint1(__pack_half2(broadcast_var_2, broadcast_var_2));
        }
        *(uint1*)(xs_local_cast + (vec * 2)) = condval_4;
      }
      *(uint4*)(((half_t*)xs) + ((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast + 0);
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      half_t broadcast_var_3 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
      uint4 condval_5;
      if ((((int)blockIdx.x) < 32)) {
        condval_5 = *(uint4*)(W0 + ((((((int)blockIdx.x) * 6144) + (i_2 * 3072)) + ((((int)threadIdx.x) >> 2) * 96)) + ((((int)threadIdx.x) & 3) * 8)));
      } else {
        condval_5 = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((half_t*)ws) + ((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_5;
    }
  } else {
    int condval_6;
    if ((((int)blockIdx.x) < 32)) {
      condval_6 = 0;
    } else {
      int condval_7;
      if ((((int)blockIdx.x) < 64)) {
        condval_7 = 1;
      } else {
        int condval_8;
        if ((((int)blockIdx.x) < 96)) {
          condval_8 = 2;
        } else {
          condval_8 = 3;
        }
        condval_7 = condval_8;
      }
      condval_6 = condval_7;
    }
    if (condval_6 == 1) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        half_t X1_local_cast_3[8];
        half_t xs_local_cast_2[8];
        half_t broadcast_var_4 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
        uint4 condval_9;
        if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_3) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
          condval_9 = *(uint4*)(X1 + ((((((int64_t)blockIdx.y) * (int64_t)6144) + (((int64_t)i_3) * (int64_t)3072)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)96)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)));
        } else {
          condval_9 = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
        }
        *(uint4*)(X1_local_cast_3 + 0) = condval_9;
        for (int vec_1 = 0; vec_1 < 4; ++vec_1) {
          half_t broadcast_var_5 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint1 condval_10;
          if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_3) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
            condval_10 = *(uint1*)(X1_local_cast_3 + (vec_1 * 2));
          } else {
            condval_10 = make_uint1(__pack_half2(broadcast_var_5, broadcast_var_5));
          }
          *(uint1*)(xs_local_cast_2 + (vec_1 * 2)) = condval_10;
        }
        *(uint4*)(((half_t*)xs) + ((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast_2 + 0);
      }
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        half_t broadcast_var_6 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
        uint4 condval_11;
        if (((((int)blockIdx.x) < 64) && (32 <= ((int)blockIdx.x)))) {
          condval_11 = *(uint4*)(W1 + (((((((int)blockIdx.x) * 6144) + (i_4 * 3072)) + ((((int)threadIdx.x) >> 2) * 96)) + ((((int)threadIdx.x) & 3) * 8)) - 196608));
        } else {
          condval_11 = make_uint4(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
        }
        *(uint4*)(((half_t*)ws) + ((((i_4 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_11;
      }
    } else {
      int condval_12;
      if ((((int)blockIdx.x) < 32)) {
        condval_12 = 0;
      } else {
        int condval_13;
        if ((((int)blockIdx.x) < 64)) {
          condval_13 = 1;
        } else {
          int condval_14;
          if ((((int)blockIdx.x) < 96)) {
            condval_14 = 2;
          } else {
            condval_14 = 3;
          }
          condval_13 = condval_14;
        }
        condval_12 = condval_13;
      }
      if (condval_12 == 2) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          half_t X2_local_cast_5[8];
          half_t xs_local_cast_4[8];
          half_t broadcast_var_7 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint4 condval_15;
          if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_5) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
            condval_15 = *(uint4*)(X2 + ((((((int64_t)blockIdx.y) * (int64_t)16384) + (((int64_t)i_5) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)));
          } else {
            condval_15 = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
          }
          *(uint4*)(X2_local_cast_5 + 0) = condval_15;
          for (int vec_2 = 0; vec_2 < 4; ++vec_2) {
            half_t broadcast_var_8 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
            uint1 condval_16;
            if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_5) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
              condval_16 = *(uint1*)(X2_local_cast_5 + (vec_2 * 2));
            } else {
              condval_16 = make_uint1(__pack_half2(broadcast_var_8, broadcast_var_8));
            }
            *(uint1*)(xs_local_cast_4 + (vec_2 * 2)) = condval_16;
          }
          *(uint4*)(((half_t*)xs) + ((((i_5 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast_4 + 0);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          half_t broadcast_var_9 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint4 condval_17;
          if (((((int)blockIdx.x) < 96) && (64 <= ((int)blockIdx.x)))) {
            condval_17 = *(uint4*)(W2 + (((((((int)blockIdx.x) * 16384) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 2) * 256)) + ((((int)threadIdx.x) & 3) * 8)) - 1048576));
          } else {
            condval_17 = make_uint4(__pack_half2(broadcast_var_9, broadcast_var_9), __pack_half2(broadcast_var_9, broadcast_var_9), __pack_half2(broadcast_var_9, broadcast_var_9), __pack_half2(broadcast_var_9, broadcast_var_9));
          }
          *(uint4*)(((half_t*)ws) + ((((i_6 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_17;
        }
      } else {
        #pragma unroll
        for (int i_7 = 0; i_7 < 2; ++i_7) {
          half_t X3_local_cast_7[8];
          half_t xs_local_cast_6[8];
          half_t broadcast_var_10 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint4 condval_18;
          if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_7) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
            condval_18 = *(uint4*)(X3 + ((((((int64_t)blockIdx.y) * (int64_t)4096) + (((int64_t)i_7) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)));
          } else {
            condval_18 = make_uint4(__pack_half2(broadcast_var_10, broadcast_var_10), __pack_half2(broadcast_var_10, broadcast_var_10), __pack_half2(broadcast_var_10, broadcast_var_10), __pack_half2(broadcast_var_10, broadcast_var_10));
          }
          *(uint4*)(X3_local_cast_7 + 0) = condval_18;
          for (int vec_3 = 0; vec_3 < 4; ++vec_3) {
            half_t broadcast_var_11 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
            uint1 condval_19;
            if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_7) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
              condval_19 = *(uint1*)(X3_local_cast_7 + (vec_3 * 2));
            } else {
              condval_19 = make_uint1(__pack_half2(broadcast_var_11, broadcast_var_11));
            }
            *(uint1*)(xs_local_cast_6 + (vec_3 * 2)) = condval_19;
          }
          *(uint4*)(((half_t*)xs) + ((((i_7 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast_6 + 0);
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          half_t broadcast_var_12 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint4 condval_20;
          if ((96 <= ((int)blockIdx.x))) {
            condval_20 = *(uint4*)(W3 + (((((((int)blockIdx.x) * 4096) + (i_8 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 8)) - 393216));
          } else {
            condval_20 = make_uint4(__pack_half2(broadcast_var_12, broadcast_var_12), __pack_half2(broadcast_var_12, broadcast_var_12), __pack_half2(broadcast_var_12, broadcast_var_12), __pack_half2(broadcast_var_12, broadcast_var_12));
          }
          *(uint4*)(((half_t*)ws) + ((((i_8 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_20;
        }
      }
    }
  }
  for (int k = 0; k < 7; ++k) {
    __syncthreads();
    int condval_21;
    if ((((int)blockIdx.x) < 32)) {
      condval_21 = 0;
    } else {
      int condval_22;
      if ((((int)blockIdx.x) < 64)) {
        condval_22 = 1;
      } else {
        int condval_23;
        if ((((int)blockIdx.x) < 96)) {
          condval_23 = 2;
        } else {
          condval_23 = 3;
        }
        condval_22 = condval_23;
      }
      condval_21 = condval_22;
    }
    if (condval_21 == 0) {
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        half_t X0_local_cast_9[8];
        half_t xs_local_cast_8[8];
        half_t broadcast_var_13 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
        uint4 condval_24;
        if (((k < 2) && ((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_9) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows))) {
          condval_24 = *(uint4*)(X0 + ((((((((int64_t)blockIdx.y) * (int64_t)6144) + (((int64_t)i_9) * (int64_t)3072)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)96)) + (((int64_t)k) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32));
        } else {
          condval_24 = make_uint4(__pack_half2(broadcast_var_13, broadcast_var_13), __pack_half2(broadcast_var_13, broadcast_var_13), __pack_half2(broadcast_var_13, broadcast_var_13), __pack_half2(broadcast_var_13, broadcast_var_13));
        }
        *(uint4*)(X0_local_cast_9 + 0) = condval_24;
        for (int vec_4 = 0; vec_4 < 4; ++vec_4) {
          half_t broadcast_var_14 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint1 condval_25;
          if ((((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_9) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows) && (k < 2))) {
            condval_25 = *(uint1*)(X0_local_cast_9 + (vec_4 * 2));
          } else {
            condval_25 = make_uint1(__pack_half2(broadcast_var_14, broadcast_var_14));
          }
          *(uint1*)(xs_local_cast_8 + (vec_4 * 2)) = condval_25;
        }
        *(uint4*)(((half_t*)xs) + (((((((k + 1) & 1) * 2048) + (i_9 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast_8 + 0);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        half_t broadcast_var_15 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
        uint4 condval_26;
        if (((((int)blockIdx.x) < 32) && (k < 2))) {
          condval_26 = *(uint4*)(W0 + ((((((((int)blockIdx.x) * 6144) + (i_10 * 3072)) + ((((int)threadIdx.x) >> 2) * 96)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
        } else {
          condval_26 = make_uint4(__pack_half2(broadcast_var_15, broadcast_var_15), __pack_half2(broadcast_var_15, broadcast_var_15), __pack_half2(broadcast_var_15, broadcast_var_15), __pack_half2(broadcast_var_15, broadcast_var_15));
        }
        *(uint4*)(((half_t*)ws) + (((((((k + 1) & 1) * 2048) + (i_10 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_26;
      }
    } else {
      int condval_27;
      if ((((int)blockIdx.x) < 32)) {
        condval_27 = 0;
      } else {
        int condval_28;
        if ((((int)blockIdx.x) < 64)) {
          condval_28 = 1;
        } else {
          int condval_29;
          if ((((int)blockIdx.x) < 96)) {
            condval_29 = 2;
          } else {
            condval_29 = 3;
          }
          condval_28 = condval_29;
        }
        condval_27 = condval_28;
      }
      if (condval_27 == 1) {
        #pragma unroll
        for (int i_11 = 0; i_11 < 2; ++i_11) {
          half_t X1_local_cast_11[8];
          half_t xs_local_cast_10[8];
          half_t broadcast_var_16 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint4 condval_30;
          if (((k < 2) && ((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_11) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows))) {
            condval_30 = *(uint4*)(X1 + ((((((((int64_t)blockIdx.y) * (int64_t)6144) + (((int64_t)i_11) * (int64_t)3072)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)96)) + (((int64_t)k) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32));
          } else {
            condval_30 = make_uint4(__pack_half2(broadcast_var_16, broadcast_var_16), __pack_half2(broadcast_var_16, broadcast_var_16), __pack_half2(broadcast_var_16, broadcast_var_16), __pack_half2(broadcast_var_16, broadcast_var_16));
          }
          *(uint4*)(X1_local_cast_11 + 0) = condval_30;
          for (int vec_5 = 0; vec_5 < 4; ++vec_5) {
            half_t broadcast_var_17 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
            uint1 condval_31;
            if ((((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_11) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows) && (k < 2))) {
              condval_31 = *(uint1*)(X1_local_cast_11 + (vec_5 * 2));
            } else {
              condval_31 = make_uint1(__pack_half2(broadcast_var_17, broadcast_var_17));
            }
            *(uint1*)(xs_local_cast_10 + (vec_5 * 2)) = condval_31;
          }
          *(uint4*)(((half_t*)xs) + (((((((k + 1) & 1) * 2048) + (i_11 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast_10 + 0);
        }
        #pragma unroll
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          half_t broadcast_var_18 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
          uint4 condval_32;
          if ((((((int)blockIdx.x) < 64) && (32 <= ((int)blockIdx.x))) && (k < 2))) {
            condval_32 = *(uint4*)(W1 + ((((((((int)blockIdx.x) * 6144) + (i_12 * 3072)) + ((((int)threadIdx.x) >> 2) * 96)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) - 196576));
          } else {
            condval_32 = make_uint4(__pack_half2(broadcast_var_18, broadcast_var_18), __pack_half2(broadcast_var_18, broadcast_var_18), __pack_half2(broadcast_var_18, broadcast_var_18), __pack_half2(broadcast_var_18, broadcast_var_18));
          }
          *(uint4*)(((half_t*)ws) + (((((((k + 1) & 1) * 2048) + (i_12 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_32;
        }
      } else {
        int condval_33;
        if ((((int)blockIdx.x) < 32)) {
          condval_33 = 0;
        } else {
          int condval_34;
          if ((((int)blockIdx.x) < 64)) {
            condval_34 = 1;
          } else {
            int condval_35;
            if ((((int)blockIdx.x) < 96)) {
              condval_35 = 2;
            } else {
              condval_35 = 3;
            }
            condval_34 = condval_35;
          }
          condval_33 = condval_34;
        }
        if (condval_33 == 2) {
          #pragma unroll
          for (int i_13 = 0; i_13 < 2; ++i_13) {
            half_t X2_local_cast_13[8];
            half_t xs_local_cast_12[8];
            half_t broadcast_var_19 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
            uint4 condval_36;
            if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_13) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
              condval_36 = *(uint4*)(X2 + ((((((((int64_t)blockIdx.y) * (int64_t)16384) + (((int64_t)i_13) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)256)) + (((int64_t)k) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32));
            } else {
              condval_36 = make_uint4(__pack_half2(broadcast_var_19, broadcast_var_19), __pack_half2(broadcast_var_19, broadcast_var_19), __pack_half2(broadcast_var_19, broadcast_var_19), __pack_half2(broadcast_var_19, broadcast_var_19));
            }
            *(uint4*)(X2_local_cast_13 + 0) = condval_36;
            for (int vec_6 = 0; vec_6 < 4; ++vec_6) {
              half_t broadcast_var_20 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
              uint1 condval_37;
              if (((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_13) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows)) {
                condval_37 = *(uint1*)(X2_local_cast_13 + (vec_6 * 2));
              } else {
                condval_37 = make_uint1(__pack_half2(broadcast_var_20, broadcast_var_20));
              }
              *(uint1*)(xs_local_cast_12 + (vec_6 * 2)) = condval_37;
            }
            *(uint4*)(((half_t*)xs) + (((((((k + 1) & 1) * 2048) + (i_13 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast_12 + 0);
          }
          #pragma unroll
          for (int i_14 = 0; i_14 < 2; ++i_14) {
            half_t broadcast_var_21 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
            uint4 condval_38;
            if (((((int)blockIdx.x) < 96) && (64 <= ((int)blockIdx.x)))) {
              condval_38 = *(uint4*)(W2 + ((((((((int)blockIdx.x) * 16384) + (i_14 * 8192)) + ((((int)threadIdx.x) >> 2) * 256)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) - 1048544));
            } else {
              condval_38 = make_uint4(__pack_half2(broadcast_var_21, broadcast_var_21), __pack_half2(broadcast_var_21, broadcast_var_21), __pack_half2(broadcast_var_21, broadcast_var_21), __pack_half2(broadcast_var_21, broadcast_var_21));
            }
            *(uint4*)(((half_t*)ws) + (((((((k + 1) & 1) * 2048) + (i_14 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_38;
          }
        } else {
          #pragma unroll
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            half_t X3_local_cast_15[8];
            half_t xs_local_cast_14[8];
            half_t broadcast_var_22 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
            uint4 condval_39;
            if (((k < 1) && ((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_15) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows))) {
              condval_39 = *(uint4*)(X3 + ((((((((int64_t)blockIdx.y) * (int64_t)4096) + (((int64_t)i_15) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)64)) + (((int64_t)k) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)) + (int64_t)32));
            } else {
              condval_39 = make_uint4(__pack_half2(broadcast_var_22, broadcast_var_22), __pack_half2(broadcast_var_22, broadcast_var_22), __pack_half2(broadcast_var_22, broadcast_var_22), __pack_half2(broadcast_var_22, broadcast_var_22));
            }
            *(uint4*)(X3_local_cast_15 + 0) = condval_39;
            for (int vec_7 = 0; vec_7 < 4; ++vec_7) {
              half_t broadcast_var_23 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
              uint1 condval_40;
              if ((((((((int64_t)blockIdx.y) * (int64_t)64) + (((int64_t)i_15) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) < rows) && (k < 1))) {
                condval_40 = *(uint1*)(X3_local_cast_15 + (vec_7 * 2));
              } else {
                condval_40 = make_uint1(__pack_half2(broadcast_var_23, broadcast_var_23));
              }
              *(uint1*)(xs_local_cast_14 + (vec_7 * 2)) = condval_40;
            }
            *(uint4*)(((half_t*)xs) + (((((((k + 1) & 1) * 2048) + (i_15 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(xs_local_cast_14 + 0);
          }
          #pragma unroll
          for (int i_16 = 0; i_16 < 2; ++i_16) {
            half_t broadcast_var_24 = __float2half_rn(0x0p+0f/*0.000000e+00*/);
            uint4 condval_41;
            if (((96 <= ((int)blockIdx.x)) && (k < 1))) {
              condval_41 = *(uint4*)(W3 + ((((((((int)blockIdx.x) * 4096) + (i_16 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) - 393184));
            } else {
              condval_41 = make_uint4(__pack_half2(broadcast_var_24, broadcast_var_24), __pack_half2(broadcast_var_24, broadcast_var_24), __pack_half2(broadcast_var_24, broadcast_var_24), __pack_half2(broadcast_var_24, broadcast_var_24));
            }
            *(uint4*)(((half_t*)ws) + (((((((k + 1) & 1) * 2048) + (i_16 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_41;
          }
        }
      }
    }
    __syncthreads();
    int condval_43;
    if ((((int)blockIdx.x) < 32)) {
      condval_43 = 0;
    } else {
      int condval_44;
      if ((((int)blockIdx.x) < 64)) {
        condval_44 = 1;
      } else {
        int condval_45;
        if ((((int)blockIdx.x) < 96)) {
          condval_45 = 2;
        } else {
          condval_45 = 3;
        }
        condval_44 = condval_45;
      }
      condval_43 = condval_44;
    }
    int condval_42;
    if ((condval_43 == 0)) {
      condval_42 = 96;
    } else {
      int condval_47;
      if ((((int)blockIdx.x) < 32)) {
        condval_47 = 0;
      } else {
        int condval_48;
        if ((((int)blockIdx.x) < 64)) {
          condval_48 = 1;
        } else {
          int condval_49;
          if ((((int)blockIdx.x) < 96)) {
            condval_49 = 2;
          } else {
            condval_49 = 3;
          }
          condval_48 = condval_49;
        }
        condval_47 = condval_48;
      }
      int condval_46;
      if ((condval_47 == 1)) {
        condval_46 = 96;
      } else {
        int condval_51;
        if ((((int)blockIdx.x) < 32)) {
          condval_51 = 0;
        } else {
          int condval_52;
          if ((((int)blockIdx.x) < 64)) {
            condval_52 = 1;
          } else {
            int condval_53;
            if ((((int)blockIdx.x) < 96)) {
              condval_53 = 2;
            } else {
              condval_53 = 3;
            }
            condval_52 = condval_53;
          }
          condval_51 = condval_52;
        }
        int condval_50;
        if ((condval_51 == 2)) {
          condval_50 = 256;
        } else {
          condval_50 = 64;
        }
        condval_46 = condval_50;
      }
      condval_42 = condval_46;
    }
    if ((k * 32) < condval_42) {
      {
        half_t A_local[32];
        half_t B_local[8];
        for (int ki = 0; ki < 2; ++ki) {
          #pragma unroll
          for (int i_17 = 0; i_17 < 4; ++i_17) {
            decode::ptx_ldmatrix_x4((&(((half_t*)xs)[((((((k & 1) * 2048) + (i_17 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_17 * 8)])));
          }
          decode::ptx_ldmatrix_x4((&(((half_t*)ws)[((((((k & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[0])));
          for (int i_18 = 0; i_18 < 4; ++i_18) {
            decode_mma(reinterpret_cast<float*>(acc + (i_18 * 8)), reinterpret_cast<const unsigned*>(A_local + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
            decode_mma(reinterpret_cast<float*>(acc + ((i_18 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local + 4));
          }
        }
      }
    }
  }
  int condval_55;
  if ((((int)blockIdx.x) < 32)) {
    condval_55 = 0;
  } else {
    int condval_56;
    if ((((int)blockIdx.x) < 64)) {
      condval_56 = 1;
    } else {
      int condval_57;
      if ((((int)blockIdx.x) < 96)) {
        condval_57 = 2;
      } else {
        condval_57 = 3;
      }
      condval_56 = condval_57;
    }
    condval_55 = condval_56;
  }
  int condval_54;
  if ((condval_55 == 0)) {
    condval_54 = 96;
  } else {
    int condval_59;
    if ((((int)blockIdx.x) < 32)) {
      condval_59 = 0;
    } else {
      int condval_60;
      if ((((int)blockIdx.x) < 64)) {
        condval_60 = 1;
      } else {
        int condval_61;
        if ((((int)blockIdx.x) < 96)) {
          condval_61 = 2;
        } else {
          condval_61 = 3;
        }
        condval_60 = condval_61;
      }
      condval_59 = condval_60;
    }
    int condval_58;
    if ((condval_59 == 1)) {
      condval_58 = 96;
    } else {
      int condval_63;
      if ((((int)blockIdx.x) < 32)) {
        condval_63 = 0;
      } else {
        int condval_64;
        if ((((int)blockIdx.x) < 64)) {
          condval_64 = 1;
        } else {
          int condval_65;
          if ((((int)blockIdx.x) < 96)) {
            condval_65 = 2;
          } else {
            condval_65 = 3;
          }
          condval_64 = condval_65;
        }
        condval_63 = condval_64;
      }
      int condval_62;
      if ((condval_63 == 2)) {
        condval_62 = 256;
      } else {
        condval_62 = 64;
      }
      condval_58 = condval_62;
    }
    condval_54 = condval_58;
  }
  if (224 < condval_54) {
    {
      half_t A_local_1[32];
      half_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        #pragma unroll
        for (int i_19 = 0; i_19 < 4; ++i_19) {
          decode::ptx_ldmatrix_x4((&(((half_t*)xs)[(((((i_19 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), (&(A_local_1[(i_19 * 8)])));
        }
        decode::ptx_ldmatrix_x4((&(((half_t*)ws)[((((((((int)threadIdx.x) >> 4) * 256) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), (&(B_local_1[0])));
        for (int i_20 = 0; i_20 < 4; ++i_20) {
          decode_mma(reinterpret_cast<float*>(acc + (i_20 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
          decode_mma(reinterpret_cast<float*>(acc + ((i_20 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 4));
        }
      }
    }
  }
  int condval_66;
  if ((((int)blockIdx.x) < 32)) {
    condval_66 = 0;
  } else {
    int condval_67;
    if ((((int)blockIdx.x) < 64)) {
      condval_67 = 1;
    } else {
      int condval_68;
      if ((((int)blockIdx.x) < 96)) {
        condval_68 = 2;
      } else {
        condval_68 = 3;
      }
      condval_67 = condval_68;
    }
    condval_66 = condval_67;
  }
  if (condval_66 == 0) {
    #pragma unroll
    for (int i_21 = 0; i_21 < 16; ++i_21) {
      float Y_local_cast_16[2];
      if ((((((((int64_t)blockIdx.y) * (int64_t)64) + ((((int64_t)i_21) >> (int64_t)2) * (int64_t)16)) + ((((int64_t)i_21) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) < rows) && (((int)blockIdx.x) < 32)) {
        *(float2*)(Y_local_cast_16 + 0) = *(float2*)(acc + (i_21 * 2));
        if ((((((((int64_t)blockIdx.y) * (int64_t)16) + ((((int64_t)i_21) >> (int64_t)2) * (int64_t)4)) + ((((int64_t)i_21) & (int64_t)1) * (int64_t)2)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)4)) - rows) < (int64_t)0) {
          *(float2*)(Y + ((((((((((int64_t)blockIdx.y) * (int64_t)131072) + ((((int64_t)i_21) >> (int64_t)2) * (int64_t)32768)) + ((((int64_t)i_21) & (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)i_21) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(Y_local_cast_16 + 0);
        }
      }
    }
  } else {
    int condval_69;
    if ((((int)blockIdx.x) < 32)) {
      condval_69 = 0;
    } else {
      int condval_70;
      if ((((int)blockIdx.x) < 64)) {
        condval_70 = 1;
      } else {
        int condval_71;
        if ((((int)blockIdx.x) < 96)) {
          condval_71 = 2;
        } else {
          condval_71 = 3;
        }
        condval_70 = condval_71;
      }
      condval_69 = condval_70;
    }
    if (condval_69 == 1) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 16; ++i_22) {
        float Y_local_cast_17[2];
        if ((((((((int64_t)blockIdx.y) * (int64_t)64) + ((((int64_t)i_22) >> (int64_t)2) * (int64_t)16)) + ((((int64_t)i_22) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) < rows) && (((int)blockIdx.x) < 64)) {
          *(float2*)(Y_local_cast_17 + 0) = *(float2*)(acc + (i_22 * 2));
          if (((((((((int64_t)blockIdx.y) * (int64_t)64) + ((((int64_t)i_22) >> (int64_t)2) * (int64_t)16)) + ((((int64_t)i_22) & (int64_t)1) * (int64_t)8)) + (((int64_t)((int)blockIdx.x)) >> (int64_t)5)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) - (rows * (int64_t)3)) < (int64_t)1) {
            *(float2*)(Y + ((((((((((((int64_t)blockIdx.y) * (int64_t)131072) + ((((int64_t)i_22) >> (int64_t)2) * (int64_t)32768)) + ((((int64_t)i_22) & (int64_t)1) * (int64_t)16384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2048)) + (rows * (int64_t)2048)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)i_22) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)2048)) = *(float2*)(Y_local_cast_17 + 0);
          }
        }
      }
    } else {
      int condval_72;
      if ((((int)blockIdx.x) < 32)) {
        condval_72 = 0;
      } else {
        int condval_73;
        if ((((int)blockIdx.x) < 64)) {
          condval_73 = 1;
        } else {
          int condval_74;
          if ((((int)blockIdx.x) < 96)) {
            condval_74 = 2;
          } else {
            condval_74 = 3;
          }
          condval_73 = condval_74;
        }
        condval_72 = condval_73;
      }
      if (condval_72 == 2) {
        #pragma unroll
        for (int i_23 = 0; i_23 < 16; ++i_23) {
          float Y_local_cast_18[2];
          if ((((((((int64_t)blockIdx.y) * (int64_t)64) + ((((int64_t)i_23) >> (int64_t)2) * (int64_t)16)) + ((((int64_t)i_23) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) < rows) && (((int)blockIdx.x) < 96)) {
            *(float2*)(Y_local_cast_18 + 0) = *(float2*)(acc + (i_23 * 2));
            if ((((((((int64_t)blockIdx.y) * (int64_t)32) + ((((int64_t)i_23) >> (int64_t)2) * (int64_t)8)) + ((((int64_t)i_23) & (int64_t)1) * (int64_t)4)) + (((((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)32) + ((int64_t)((int)blockIdx.x))) >> (int64_t)6)) - rows) < (int64_t)1) {
              *(float2*)(Y + ((((((((((((int64_t)blockIdx.y) * (int64_t)131072) + ((((int64_t)i_23) >> (int64_t)2) * (int64_t)32768)) + ((((int64_t)i_23) & (int64_t)1) * (int64_t)16384)) + (rows * (int64_t)4096)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)i_23) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)4096)) = *(float2*)(Y_local_cast_18 + 0);
            }
          }
        }
      } else {
        #pragma unroll
        for (int i_24 = 0; i_24 < 16; ++i_24) {
          float Y_local_cast_19[2];
          if (((((((int64_t)blockIdx.y) * (int64_t)64) + ((((int64_t)i_24) >> (int64_t)2) * (int64_t)16)) + ((((int64_t)i_24) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) < rows) {
            *(float2*)(Y_local_cast_19 + 0) = *(float2*)(acc + (i_24 * 2));
            *(float2*)(Y + ((((((((((((int64_t)blockIdx.y) * (int64_t)131072) + ((((int64_t)i_24) >> (int64_t)2) * (int64_t)32768)) + ((((int64_t)i_24) & (int64_t)1) * (int64_t)16384)) + (rows * (int64_t)6144)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)i_24) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) - (int64_t)6144)) = *(float2*)(Y_local_cast_19 + 0);
          }
        }
      }
    }
  }
}


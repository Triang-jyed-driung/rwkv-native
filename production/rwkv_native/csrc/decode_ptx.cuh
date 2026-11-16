// SPDX-License-Identifier: MIT
// Copyright (c) Tile-AI. Adapted from TileLang CUDA copy.h, ldsm.h and mma.h.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
using half_t=__half;
using uchar=unsigned char;
namespace decode {
__device__ __forceinline__ void cp_async_commit(){asm volatile("cp.async.commit_group;"::);}
template<int N> __device__ __forceinline__ void cp_async_wait(){
 if constexpr(N==0)asm volatile("cp.async.wait_all;"::);
 else asm volatile("cp.async.wait_group %0;"::"n"(N));
}
template<int N> __device__ __forceinline__ void cp_async_gs_conditional(const void* smem,const void* global,bool valid){
 static_assert(N==8 || N==16);
 unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(smem));int bytes=valid?N:0;
 if constexpr(N==16)asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;"::"r"(address),"l"(global),"n"(N),"r"(bytes));
 else asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;"::"r"(address),"l"(global),"n"(N),"r"(bytes));
}
__device__ __forceinline__ void ptx_ldmatrix_x4(const void* smem,void* local){
 unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(smem));int32_t* out=reinterpret_cast<int32_t*>(local);
 asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(out[0]),"=r"(out[1]),"=r"(out[2]),"=r"(out[3]):"r"(address));
}
}
__device__ __forceinline__ void decode_mma(float* c,const unsigned* a,const unsigned* b){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
 : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

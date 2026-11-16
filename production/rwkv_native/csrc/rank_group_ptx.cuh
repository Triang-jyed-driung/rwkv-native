// SPDX-License-Identifier: MIT
// Required packing helper from TileLang common.h; no optional dtype dispatch.
#pragma once
#include "decode_ptx.cuh"
__device__ __forceinline__ unsigned __pack_half2(half_t x,half_t y){
 return unsigned(__half_as_ushort(x))|(unsigned(__half_as_ushort(y))<<16);
}

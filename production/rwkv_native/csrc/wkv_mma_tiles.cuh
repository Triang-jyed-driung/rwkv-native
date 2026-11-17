#pragma once
#include <initializer_list>
// Copyright (c) 2024, Johan Sokrates Wind

#include <cuda_bf16.h>
#include <cuda_fp16.h>

using Half = __half;
using Half2 = __half2;
using uint = unsigned int;
__device__ constexpr int tile_index(int row, int col) { return row * 16 + (col ^ ((row & 4) * 2)); }
__device__ inline float to_float(const Half &u) { return __half2float(u); }
__device__ inline Half to_half(const float &u) { return __float2half_rn(u); }
__device__ inline float2 to_float2(const Half2 &u) { return __half22float2(u); }
__device__ inline float2 to_float2(const float2 &u) { return u; }
__device__ inline Half2 to_half2(const float2 &u) { return __float22half2_rn(u); }
__device__ inline uint &as_uint(const Half2 &x) { return *((uint *)(&x)); }
__device__ inline uint __smem(const void *x) { return __cvta_generic_to_shared(x); }

__device__ void commit_copies() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ void wait_copies() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ void operator*=(float2 &a, const float2 &b) {
    a.x *= b.x;
    a.y *= b.y;
}
__device__ void operator+=(float2 &a, const float2 &b) {
    a.x += b.x;
    a.y += b.y;
}
__device__ float2 operator+(const float2 &a, const float2 &b) { return {a.x + b.x, a.y + b.y}; }
__device__ float2 operator*(const float2 &a, const float2 &b) { return {a.x * b.x, a.y * b.y}; }

// SharedTile: swizzled FP16 matrix (16 x 16).
// HalfTile: warp-distributed FP16 MMA operand.
// FloatTile: warp-distributed FP32 accumulator.
// operator% means A * transpose(B), not elementwise multiplication.
struct SharedTile;
struct HalfTile;
struct FloatTile;

struct GlobalBf16Tile {
    __nv_bfloat16 *ga;
    int stride;
    __device__ GlobalBf16Tile(__nv_bfloat16 *ga_, int stride_) : ga(ga_), stride(stride_) {}
    __device__ GlobalBf16Tile &operator=(const HalfTile &);
};
struct TransposedSharedTile {
    SharedTile *st;
};

struct __align__(16) SharedTile {
    Half data[16 * 16];
    __device__ SharedTile() {}
    __device__ SharedTile(const HalfTile &o) { *this = o; }
    __device__ SharedTile &operator=(const GlobalBf16Tile &);
    __device__ SharedTile &operator=(const HalfTile &);
    __device__ TransposedSharedTile t() { return TransposedSharedTile{this}; }
};
struct Product {
    const HalfTile *a, *b;
};
struct HalfTile {
    Half2 data[4];
    __device__ HalfTile() {}
    __device__ void zero_() { data[0] = data[1] = data[2] = data[3] = to_half2({0.f, 0.f}); }
    __device__ HalfTile(const SharedTile &o) { *this = o; }
    __device__ HalfTile(const TransposedSharedTile &o) { *this = o; }
    __device__ HalfTile(const FloatTile &o) { *this = o; }
    __device__ HalfTile &operator=(const SharedTile &);
    __device__ HalfTile &operator=(const TransposedSharedTile &);
    __device__ HalfTile &operator=(const FloatTile &fa);
    __device__ HalfTile &operator=(const GlobalBf16Tile &);
};
struct FloatTile {
    union {
        float2 data[4];
        float fdata[8];
    };
    __device__ void zero_() { data[0] = data[1] = data[2] = data[3] = {0.f, 0.f}; }
    __device__ FloatTile() {}
    __device__ FloatTile(const FloatTile &o) {
        for (int i = 0; i < 4; i++)
            data[i] = o.data[i];
    }
    __device__ FloatTile(const HalfTile &r) { *this = r; }
    __device__ FloatTile(const Product &p) { *this = p; }
    __device__ FloatTile &operator=(const Product &);
    __device__ FloatTile &operator=(const HalfTile &);
    __device__ FloatTile &operator+=(const Product &);
    __device__ FloatTile &operator+=(const FloatTile &o) {
        for (int i = 0; i < 4; i++)
            data[i] += o.data[i];
        return *this;
    }
};

__device__ void load(SharedTile &sa, __nv_bfloat16 *ga, int stride) {
    int i = threadIdx.x % 32 / 2, j = threadIdx.x % 2;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(__smem(&sa.data[tile_index(i, j * 8)])),
        "l"(ga + i * stride + j * 8)
        : "memory");
}
__device__ void convert_stage(SharedTile &sa) {
    for (int z = threadIdx.x % 32; z < 128; z += 32) {
        float2 v = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(sa.data)[z]);
        reinterpret_cast<__half2 *>(sa.data)[z] = __float22half2_rn(v);
    }
}

__device__ void load(HalfTile &ra, const SharedTile &sa) {
    int i = threadIdx.x % 8, j = threadIdx.x % 32 / 16, k = threadIdx.x / 8 % 2;
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(as_uint(ra.data[0])), "=r"(as_uint(ra.data[1])), "=r"(as_uint(ra.data[2])),
                   "=r"(as_uint(ra.data[3]))
                 : "r"(__smem(&sa.data[tile_index(i + k * 8, j * 8)])));
}
__device__ void loadT(HalfTile &ra, const SharedTile &sa) {
    int i = threadIdx.x % 8, j = threadIdx.x % 32 / 16, k = threadIdx.x / 8 % 2;
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(as_uint(ra.data[0])), "=r"(as_uint(ra.data[1])), "=r"(as_uint(ra.data[2])),
                   "=r"(as_uint(ra.data[3]))
                 : "r"(__smem(&sa.data[tile_index(i + j * 8, k * 8)])));
}

__device__ static inline void __m16n8k16(float2 &d0, float2 &d1, const Half2 &a0, const Half2 &a1,
                                         const Half2 &a2, const Half2 &a3, const Half2 &b0,
                                         const Half2 &b1, const float2 &c0, const float2 &c1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, "
                 "%7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=f"(d0.x), "=f"(d0.y), "=f"(d1.x), "=f"(d1.y)
                 : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(a2)), "r"(as_uint(a3)),
                   "r"(as_uint(b0)), "r"(as_uint(b1)), "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}
__device__ void mma(FloatTile &rd, const HalfTile &ra, const HalfTile &rb,
                    const FloatTile &rc) { // d = a*b^T + c
    __m16n8k16(rd.data[0], rd.data[1], ra.data[0], ra.data[1], ra.data[2], ra.data[3], rb.data[0],
               rb.data[2], rc.data[0], rc.data[1]);
    __m16n8k16(rd.data[2], rd.data[3], ra.data[0], ra.data[1], ra.data[2], ra.data[3], rb.data[1],
               rb.data[3], rc.data[2], rc.data[3]);
}
__device__ static inline void __m16n8k16(float2 &d0, float2 &d1, const Half2 &a0, const Half2 &a1,
                                         const Half2 &a2, const Half2 &a3, const Half2 &b0,
                                         const Half2 &b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, "
                 "%7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)
                 : "r"(as_uint(a0)), "r"(as_uint(a1)), "r"(as_uint(a2)), "r"(as_uint(a3)),
                   "r"(as_uint(b0)), "r"(as_uint(b1)), "f"(d0.x), "f"(d0.y), "f"(d1.x), "f"(d1.y));
}
__device__ void mma(FloatTile &rd, const HalfTile &ra, const HalfTile &rb) { // d += a*b^T
    __m16n8k16(rd.data[0], rd.data[1], ra.data[0], ra.data[1], ra.data[2], ra.data[3], rb.data[0],
               rb.data[2]);
    __m16n8k16(rd.data[2], rd.data[3], ra.data[0], ra.data[1], ra.data[2], ra.data[3], rb.data[1],
               rb.data[3]);
}
__device__ void mm(FloatTile &rd, const HalfTile &ra, const HalfTile &rb) { // d = a*b^T
    __m16n8k16(rd.data[0], rd.data[1], ra.data[0], ra.data[1], ra.data[2], ra.data[3], rb.data[0],
               rb.data[2], {0.f, 0.f}, {0.f, 0.f});
    __m16n8k16(rd.data[2], rd.data[3], ra.data[0], ra.data[1], ra.data[2], ra.data[3], rb.data[1],
               rb.data[3], {0.f, 0.f}, {0.f, 0.f});
}

__device__ void store(const HalfTile &ra, __nv_bfloat16 *ga, int stride) {
    int i = threadIdx.x % 32 / 4, j = threadIdx.x % 4 * 2;
    for (int q = 0; q < 4; ++q) {
        int off = (i + (q % 2) * 8) * stride + j + (q / 2) * 8;
        float2 z = to_float2(ra.data[q]);
        ga[off] = __float2bfloat16_rn(z.x);
        ga[off + 1] = __float2bfloat16_rn(z.y);
    }
}
__device__ void load(HalfTile &ra, __nv_bfloat16 *ga, int stride) {
    int i = threadIdx.x % 32 / 4, j = threadIdx.x % 4 * 2;
    for (int q = 0; q < 4; ++q) {
        int off = (i + (q % 2) * 8) * stride + j + (q / 2) * 8;
        ra.data[q] = to_half2({__bfloat162float(ga[off]), __bfloat162float(ga[off + 1])});
    }
}
__device__ void store(const HalfTile &ra, SharedTile &sa) {
    int row = threadIdx.x % 32 / 4, col = threadIdx.x % 4 * 2;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<Half2 *>(&sa.data[tile_index(row + j % 2 * 8, col + j / 2 * 8)]) =
            ra.data[j];
}

__device__ void convert(HalfTile &ra, const FloatTile &fa) {
    ra.data[0] = to_half2(fa.data[0]);
    ra.data[1] = to_half2(fa.data[1]);
    ra.data[2] = to_half2(fa.data[2]);
    ra.data[3] = to_half2(fa.data[3]);
}
__device__ void convert(FloatTile &fa, const HalfTile &ra) {
    fa.data[0] = to_float2(ra.data[0]);
    fa.data[1] = to_float2(ra.data[1]);
    fa.data[2] = to_float2(ra.data[2]);
    fa.data[3] = to_float2(ra.data[3]);
}

__device__ SharedTile &SharedTile::operator=(const GlobalBf16Tile &ga) {
    load(*this, ga.ga, ga.stride);
    return *this;
}
__device__ HalfTile &HalfTile::operator=(const GlobalBf16Tile &ga) {
    load(*this, ga.ga, ga.stride);
    return *this;
}
__device__ HalfTile &HalfTile::operator=(const SharedTile &sa) {
    load(*this, sa);
    return *this;
}
__device__ SharedTile &SharedTile::operator=(const HalfTile &ra) {
    store(ra, *this);
    return *this;
}
__device__ HalfTile &HalfTile::operator=(const TransposedSharedTile &sa) {
    loadT(*this, *sa.st);
    return *this;
}
__device__ Product operator%(const HalfTile &ra, const HalfTile &rb) { return Product{&ra, &rb}; }
__device__ FloatTile &FloatTile::operator=(const Product &prod) {
    mm(*this, *prod.a, *prod.b);
    return *this;
}
__device__ FloatTile &FloatTile::operator+=(const Product &prod) {
    mma(*this, *prod.a, *prod.b);
    return *this;
}
__device__ HalfTile &HalfTile::operator=(const FloatTile &fa) {
    convert(*this, fa);
    return *this;
}
__device__ FloatTile &FloatTile::operator=(const HalfTile &ra) {
    convert(*this, ra);
    return *this;
}
__device__ GlobalBf16Tile &GlobalBf16Tile::operator=(const HalfTile &ra) {
    store(ra, this->ga, this->stride);
    return *this;
}

// Warp scans along the 16 time rows. The argument is updated in place.
template <int inclusive, int rev> __device__ FloatTile cumsumv(FloatTile &w) {
    int tid = threadIdx.x % 32, t = tid / 4;

    FloatTile ret;
    if (inclusive)
        for (int i = 0; i < 4; i++)
            ret.data[i] = w.data[i];
    else
        for (int i = 0; i < 4; i++)
            ret.data[i] = float2{0.f, 0.f};

    for (int b = 0; b < 3; b++) {
        for (int i = 0; i < 8; i++) {
            float other_w = __shfl_xor_sync(0xffffffff, w.fdata[i], 4 << b);
            if ((t >> b) % 2 == !rev)
                ret.fdata[i] += other_w;
            w.fdata[i] += other_w;
        }
    }
    for (int i : {0, 1, 4, 5}) {
        float &w0 = w.fdata[i ^ (2 * !rev)], &w1 = w.fdata[i ^ (2 * rev)];
        ret.fdata[i ^ (2 * !rev)] += w1;
        w0 += w1;
        w1 = w0;
    }
    return ret;
}

template <int inclusive, int rev> __device__ FloatTile cumprodv(FloatTile &w) {
    int tid = threadIdx.x % 32, t = tid / 4;

    FloatTile ret;
    if (inclusive)
        for (int i = 0; i < 4; i++)
            ret.data[i] = w.data[i];
    else
        for (int i = 0; i < 4; i++)
            ret.data[i] = float2{1.f, 1.f};

    for (int b = 0; b < 3; b++) {
        for (int i = 0; i < 8; i++) {
            float other_w = __shfl_xor_sync(0xffffffff, w.fdata[i], 4 << b);
            if ((t >> b) % 2 == !rev)
                ret.fdata[i] *= other_w;
            w.fdata[i] *= other_w;
        }
    }
    for (int i : {0, 1, 4, 5}) {
        float &w0 = w.fdata[i ^ (2 * !rev)], &w1 = w.fdata[i ^ (2 * rev)];
        ret.fdata[i ^ (2 * !rev)] *= w1;
        w0 *= w1;
        w1 = w0;
    }
    return ret;
}

__device__ FloatTile operator*(const FloatTile &a, const FloatTile &b) {
    FloatTile ret;
    for (int i = 0; i < 8; i++)
        ret.fdata[i] = a.fdata[i] * b.fdata[i];
    return ret;
}

template <int strict> __device__ FloatTile tril(const FloatTile &f) {
    int i0 = threadIdx.x % 32 / 4, j0 = threadIdx.x % 4 * 2;
    FloatTile ret;
    for (int k = 0; k < 8; k++) {
        int i = i0 + k / 2 % 2 * 8, j = j0 + k % 2 + k / 4 * 8;
        if (strict)
            ret.fdata[k] = (i > j ? f.fdata[k] : 0.f);
        else
            ret.fdata[k] = (i >= j ? f.fdata[k] : 0.f);
    }
    return ret;
}

template <class F> __device__ void apply_(FloatTile &tile, F f) {
    for (int i = 0; i < 8; i++)
        tile.fdata[i] = f(tile.fdata[i]);
}

__device__ Half2 transpose(Half2 a) {
    Half2 ret;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                 : "=r"(as_uint(ret))
                 : "r"(as_uint(a)));
    return ret;
}

__device__ HalfTile transpose(const HalfTile &ra) {
    HalfTile rb;
    rb.data[0] = transpose(ra.data[0]);
    rb.data[1] = transpose(ra.data[2]);
    rb.data[2] = transpose(ra.data[1]);
    rb.data[3] = transpose(ra.data[3]);
    return rb;
}

__device__ FloatTile transpose_float(const FloatTile &x) {
    FloatTile y;
    int lane = threadIdx.x % 32;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int base = (j / 4) * 2 + ((j / 2) % 2) * 4;
        int src = (lane % 4 * 2 + j % 2) * 4 + lane / 8;
        float lo = __shfl_sync(0xffffffff, x.fdata[base], src);
        float high = __shfl_sync(0xffffffff, x.fdata[base + 1], src);
        y.fdata[j] = (lane / 4) % 2 ? high : lo;
    }
    return y;
}

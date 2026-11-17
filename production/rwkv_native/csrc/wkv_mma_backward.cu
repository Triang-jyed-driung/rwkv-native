// Backward-only adapter for rwkv-native. Derived from wkv7_fast/kernel.cu.
// Copyright notices and the upstream MIT license are retained in this folder.
// The original FP32 forward and its native [B,H,chunk,K,V] checkpoints are unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
// Copyright (c) 2024, Johan Sokrates Wind

#include "wkv_mma_tiles.cuh"
#include <assert.h>
// State recurrence (state rows are value channels, columns are key channels):
// u = S * a; S = S * diag(decay) + u * b^T + v * k^T; y = S * r.
// q in this CUDA implementation denotes the public Python argument r.
// Process 16 tokens per Tensor Core block, with FP32 live forward state.
// Each pair stores its first FP16(state/128) checkpoint plus the first block's U.
// Backward reconstructs the intervening checkpoint inside the CTA and reuses U.
// The packed record is 10 KiB per 32 tokens; no extra replay kernel is launched.
// Every sequence starts its own pair. Allocation includes padding for ragged starts.
// All input shapes use the same kernel and compile-time pipeline configuration.
// Power-of-two balancing keeps reciprocal-prefix operands in FP16 range.
// FP32 accumulators are rounded directly to BF16 at the public boundary.
__device__ FloatTile scale_tile(FloatTile x, float scale) {
    for (int i = 0; i < 8; ++i)
        x.fdata[i] *= scale;
    return x;
}
__device__ FloatTile refined_product(const FloatTile &a, const FloatTile &b) {
    HalfTile ah = a, bh = b;
    FloatTile ar = ah, br = bh;
    for (int i = 0; i < 8; ++i) {
        ar.fdata[i] = (a.fdata[i] - ar.fdata[i]) * 1024.f;
        br.fdata[i] = (b.fdata[i] - br.fdata[i]) * 1024.f;
    }
    FloatTile correction = (HalfTile)ar % bh;
    correction += ah % (HalfTile)br;
    FloatTile result = scale_tile(correction, 1.f / 1024.f);
    result += ah % bh;
    return result;
}
// A^4*A^4 has nonzeros only in the left eight columns. A^8 additionally
// consumes only the upper eight reduction indices. Keep FP32 corrections.
template<bool HighK>
__device__ __forceinline__ void triangular_mma(FloatTile &d,const HalfTile &a,const HalfTile &b) {
    if constexpr(HighK) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
          : "+f"(d.fdata[0]),"+f"(d.fdata[1]),"+f"(d.fdata[2]),"+f"(d.fdata[3])
          : "r"(as_uint(a.data[2])),"r"(as_uint(a.data[3])),"r"(as_uint(b.data[2])));
    } else {
        __m16n8k16(d.data[0],d.data[1],a.data[0],a.data[1],a.data[2],a.data[3],b.data[0],b.data[2]);
    }
}
template<bool HighK>
__device__ __forceinline__ FloatTile triangular_product(const FloatTile &a,const FloatTile &b) {
    HalfTile ah=a,bh=b;FloatTile ar=ah,br=bh;
#pragma unroll
    for(int i=0;i<8;++i) {ar.fdata[i]=(a.fdata[i]-ar.fdata[i])*1024.f;br.fdata[i]=(b.fdata[i]-br.fdata[i])*1024.f;}
    FloatTile correction;correction.zero_();
    triangular_mma<HighK>(correction,HalfTile(ar),bh);
    triangular_mma<HighK>(correction,ah,HalfTile(br));
    FloatTile result=scale_tile(correction,1.f/1024.f);
    triangular_mma<HighK>(result,ah,bh);
    return result;
}
template<bool HighK>
__device__ __forceinline__ FloatTile triangular_plain(const FloatTile &a,const FloatTile &b) {
    FloatTile out; out.zero_();
    triangular_mma<HighK>(out,HalfTile(a),HalfTile(b));
    return out;
}
// Strict triangular A has A^16=0. Doubling computes the finite inverse
// polynomial. Only A^4*A^4 omits residual correction; the final inverse update
// retains correction. The fixed 16-step coop_fwd8 absolute-error gate is validated.
__device__ FloatTile unit_lower_inverse(FloatTile a) {
    FloatTile power = refined_product(a, transpose_float(a));
    FloatTile result = a;
    int row = threadIdx.x % 32 / 4, col = threadIdx.x % 4 * 2;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        if (row + (j / 2 % 2) * 8 == col + j % 2 + (j / 4) * 8)
            result.fdata[j] += 1.f;
#pragma unroll
    for (int step = 0; step < 3; ++step) {
        result += step==2 ? triangular_product<true>(result,transpose_float(power)) : refined_product(result,transpose_float(power));
        if (step < 2)
            power = step==1 ? triangular_plain<false>(power,transpose_float(power)) : refined_product(power,transpose_float(power));
    }
    return result;
}
// Reduce two triangular tiles together so they scratch block barriers.
__device__ void reduce_pair(FloatTile &a, FloatTile &b, float *scratch, bool enter = true) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (enter)
        __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j == 4 || j == 5)
            continue;
        scratch[(warp * 8 + j) * 32 + lane] = a.fdata[j];
        scratch[1024 + (warp * 8 + j) * 32 + lane] = b.fdata[j];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j == 4 || j == 5) {
            a.fdata[j] = b.fdata[j] = 0.f;
            continue;
        }
        float x = 0.f, y = 0.f;
#pragma unroll
        for (int w = 1; w < 4; ++w) {
            x += scratch[(w * 8 + j) * 32 + lane];
            y += scratch[1024 + (w * 8 + j) * 32 + lane];
        }
        a.fdata[j] = scratch[j * 32 + lane] + x;
        b.fdata[j] = scratch[1024 + j * 32 + lane] + y;
    }
}
// Each warp fetches four complete token rows, avoiding partial L1 lines.
__device__ void load_head(SharedTile *dst, const __nv_bfloat16 *src, int stride) {
    int row = threadIdx.x / 8, col = (threadIdx.x % 8) * 8;
    unsigned shared = __cvta_generic_to_shared(&dst[col / 16].data[tile_index(row, col % 16)]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(shared),
                 "l"(src + row * stride + col)
                 : "memory");
}
__device__ void store_bf16(FloatTile x, __nv_bfloat16 *dst, int stride, float scale = 1.f) {
    int row = threadIdx.x % 32 / 4, col = threadIdx.x % 4 * 2;
    for (int j = 0; j < 4; ++j) {
        int off = (row + (j % 2) * 8) * stride + col + (j / 2) * 8;
        float2 v = x.data[j];
        v.x *= scale;
        v.y *= scale;
        *reinterpret_cast<__nv_bfloat162 *>(dst + off) = __float22bfloat162_rn(v);
    }
}
__device__ void convert_scaled_stage(SharedTile &s, float scale) {
    for (int z = threadIdx.x % 32; z < 128; z += 32) {
        float2 v = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(s.data)[z]);
        v.x *= scale;
        v.y *= scale;
        reinterpret_cast<__half2 *>(s.data)[z] = __float22half2_rn(v);
    }
}
__device__ FloatTile row_sum_broadcast(const FloatTile &x) {
    float lo = x.fdata[0] + x.fdata[1] + x.fdata[4] + x.fdata[5];
    float high = x.fdata[2] + x.fdata[3] + x.fdata[6] + x.fdata[7];
    lo += __shfl_xor_sync(0xffffffff, lo, 1);
    high += __shfl_xor_sync(0xffffffff, high, 1);
    lo += __shfl_xor_sync(0xffffffff, lo, 2);
    high += __shfl_xor_sync(0xffffffff, high, 2);
    int src = (threadIdx.x % 4) * 8;
    FloatTile out;
    out.data[0] =
        out.data[1] = {__shfl_sync(0xffffffff, lo, src), __shfl_sync(0xffffffff, lo, src + 4)};
    out.data[2] =
        out.data[3] = {__shfl_sync(0xffffffff, high, src), __shfl_sync(0xffffffff, high, src + 4)};
    return out;
}
// The chunk decay is constant across time. Each warp publishes 16 FP32 values.
__device__ void publish_decay(const FloatTile &x, float *dst, int warp) {
    int lane = threadIdx.x % 32;
    if (lane < 4) {
        *reinterpret_cast<float2 *>(dst + warp * 16 + lane * 2) = x.data[0];
        *reinterpret_cast<float2 *>(dst + warp * 16 + lane * 2 + 8) = x.data[2];
    }
}
__device__ FloatTile cached_decay(const float *src, int warp) {
    int col = (threadIdx.x % 4) * 2;
    FloatTile x;
    x.data[0] = x.data[1] = *reinterpret_cast<const float2 *>(src + warp * 16 + col);
    x.data[2] = x.data[3] = *reinterpret_cast<const float2 *>(src + warp * 16 + col + 8);
    return x;
}

__device__ void store_half_checkpoint(HalfTile x, __half *dst, int stride) {
    int row = threadIdx.x % 32 / 4, col = threadIdx.x % 4 * 2;
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<__half2 *>(dst + (row + j % 2 * 8) * stride + col + j / 2 * 8) =
            x.data[j];
}

typedef __nv_bfloat16 *__restrict__ Bf16Ptr;
typedef float *__restrict__ FloatPtr;

// Fixed configuration validated on RTX 4090 (sm_89).
constexpr int kHeadSize = 64;
constexpr int kWarps = 4;
constexpr int kForwardStages = 4;
constexpr int kBackwardStages = 3;

__device__ HalfTile raw_bf16(const SharedTile &tile) {
    HalfTile x = tile;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&x.data[j]));
        x.data[j] = __float22half2_rn(f);
    }
    return x;
}


__device__ void store_gradient(FloatTile x, float *dst, int stride, float scale,
                              int t0,int T,const bool *mask) {
    int row=(threadIdx.x%32)/4,col=(threadIdx.x%4)*2;
    for(int j=0;j<4;++j) {
        int t=t0+row+(j%2)*8;
        if(t<T) {
            float2 value=x.data[j];
            int off=(row+(j%2)*8)*stride+col+(j/2)*8;
            if(!mask[t])value={0.f,0.f};
            else {value.x*=scale;value.y*=scale;}
            *reinterpret_cast<float2*>(dst+off)=value;
        }
    }
}
__global__ void backward_kernel(
    int T, int H, Bf16Ptr w_, Bf16Ptr q_, Bf16Ptr k_, Bf16Ptr v_, Bf16Ptr a_, Bf16Ptr b_,
    Bf16Ptr dy_, FloatPtr checkpoints, const float *final_state_grad, float *dw_,
    float *dq_, float *dk_, float *dv_, float *da_,
    float *db_, float *initial_state_grad, const float *ranges, const bool *mask, int original_T) {
    constexpr int C = kHeadSize, K = 16;
    int batch = blockIdx.y, head = blockIdx.x;
    // One record = 4096 half state values + 1024 half U values.
    // Padding separates pairs at ragged sequence boundaries.
    int first_chunk = batch*T/K, chunk_count = T/K;
    extern __shared__ char smem_[];
    char *smem = smem_;

    SharedTile *w_tiles = (SharedTile *)smem;
    smem += sizeof(SharedTile) * kBackwardStages * kWarps;
    SharedTile *q_tiles = (SharedTile *)smem;
    smem += sizeof(SharedTile) * kBackwardStages * kWarps;
    SharedTile *k_tiles = (SharedTile *)smem;
    smem += sizeof(SharedTile) * kBackwardStages * kWarps;
    SharedTile *v_tiles = (SharedTile *)smem;
    smem += sizeof(SharedTile) * kBackwardStages * kWarps;
    SharedTile *a_tiles = (SharedTile *)smem;
    smem += sizeof(SharedTile) * kBackwardStages * kWarps;
    SharedTile *b_tiles = (SharedTile *)smem;
    smem += sizeof(SharedTile) * kBackwardStages * kWarps;
    SharedTile *dy_tiles = (SharedTile *)smem;
    smem += sizeof(SharedTile) * kBackwardStages * kWarps;
    SharedTile *backward_cache = (SharedTile *)smem;
    smem += sizeof(SharedTile) * 24;
    char *scratch = (char *)smem;
    __shared__ SharedTile checkpoint_raw[kBackwardStages][kWarps * kWarps];
    __shared__ SharedTile replay_ut[kBackwardStages][kWarps];

    int stride = H * C;
    int warp = threadIdx.x / 32;

    // One power-of-two scale per sequence/head, independent of shape.
    // Bound the largest incoming adjoint near 8 while retaining small gradients.
    bool active = false;
    for (int t = threadIdx.x; t < original_T; t += blockDim.x)
        active |= mask[batch * original_T + t];
    if (!__syncthreads_or(active)) {
        for (int z = threadIdx.x; z < original_T * C; z += blockDim.x) {
            int off = (batch * original_T + z/C) * H*C + head*C + z%C;
            dw_[off]=dq_[off]=dk_[off]=dv_[off]=da_[off]=db_[off]=0.f;
        }
        for (int z = threadIdx.x; z < C*C; z += blockDim.x) {
            int off = (batch*H+head)*C*C+z;
            initial_state_grad[off] = final_state_grad ? final_state_grad[off] : 0.f;
        }
        return;
    }
    float largest = ranges[batch*H+head];
    float exponent = fminf(24.f, fmaxf(-16.f, floorf(log2f(8.f / fmaxf(largest, 1e-30f)))));
    float grad_scale = exp2f(exponent), inv_grad_scale = exp2f(-exponent);
    auto push = [&](int t) {
        int off = (first_chunk + t) * K * H * C + head * C + warp * 16;
        int stage = t % kBackwardStages;
        load_head(w_tiles + stage * kWarps, w_ + off - warp * 16, stride);
        load_head(q_tiles + stage * kWarps, q_ + off - warp * 16, stride);
        load_head(k_tiles + stage * kWarps, k_ + off - warp * 16, stride);
        load_head(v_tiles + stage * kWarps, v_ + off - warp * 16, stride);
        load_head(a_tiles + stage * kWarps, a_ + off - warp * 16, stride);
        load_head(b_tiles + stage * kWarps, b_ + off - warp * 16, stride);
        load_head(dy_tiles + stage * kWarps, dy_ + off - warp * 16, stride);
        for (int i = 0; i < kWarps; i++) {
            long long off2 = ((long long)(batch*H+head)*chunk_count+t)*C*C
                             + warp*16*C+i*16;
            int z = (threadIdx.x%32)*8;
            unsigned dst = __cvta_generic_to_shared(
                &checkpoint_raw[stage][warp*kWarps+i].data[tile_index(z/16,z%16)]);
            const __half *src = reinterpret_cast<const __half *>(checkpoints)+off2+(z/16)*C+z%16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),"l"(src):"memory");
        }
    };

    FloatTile dstate[kWarps];
    for (int i=0;i<kWarps;++i) {
        for (int j=0;j<8;++j) {
            int value=warp*16+(threadIdx.x%32)/4+((j/2)%2)*8;
            int key=i*16+(threadIdx.x%4)*2+j%2+(j/4)*8;
            long long off=(long long)(batch*H+head)*C*C+key*C+value;
            dstate[i].fdata[j]=final_state_grad ? final_state_grad[off]*grad_scale : 0.f;
        }
    }

    for (int t = 0; t < kBackwardStages - 1; t++) {
        if (t < chunk_count)
            push(chunk_count - 1 - t);
        commit_copies();
    }

    for (int t = chunk_count - 1; t >= 0; t--) {
        __syncthreads();
        if (t - kBackwardStages + 1 >= 0)
            push(t - kBackwardStages + 1);
        commit_copies();
        wait_copies<kBackwardStages - 1>();
        __syncthreads();
        int stage = t % kBackwardStages;
        SharedTile &sw = w_tiles[stage * kWarps + warp], &sq = q_tiles[stage * kWarps + warp],
                   &sk = k_tiles[stage * kWarps + warp], &sv = v_tiles[stage * kWarps + warp],
                   &sa = a_tiles[stage * kWarps + warp], &sb = b_tiles[stage * kWarps + warp],
                   &sdy = dy_tiles[stage * kWarps + warp];
        SharedTile *state = checkpoint_raw[stage];
        convert_scaled_stage(sdy, grad_scale);

        convert_stage(sw);
        convert_stage(sq);
        convert_stage(sk);
        convert_stage(sv);
        convert_stage(sa);
        convert_stage(sb);
        __syncwarp();
        FloatTile w = (HalfTile)sw;
        apply_(w, [](float x) { return __expf(-0.6065306597126334f / (1.f + __expf(-x))); });
        FloatTile fw = w;
        FloatTile non_incl_pref = cumprodv<0, 0>(fw);
        FloatTile incl_pref = non_incl_pref * w;
        FloatTile inv_incl_pref = incl_pref;
        apply_(inv_incl_pref, [](float x) { return 1.f / x; });

        non_incl_pref = scale_tile(non_incl_pref, 128.f);
        incl_pref = scale_tile(incl_pref, 128.f);
        inv_incl_pref = scale_tile(inv_incl_pref, 1.f / 128.f);
        HalfTile wq = (HalfTile)sq * incl_pref, kwi = (HalfTile)sk * inv_incl_pref;
        HalfTile wa = (HalfTile)sa * non_incl_pref, bwi = (HalfTile)sb * inv_incl_pref;
        HalfTile vt = sv.t();
        // Retire warp-local readers before replacing their input tiles.
        __syncwarp();
        sq = scale_tile(wq, 1.f / 128.f);
        sa = wa;
        sk = scale_tile(kwi * fw, 128.f);
        sb = scale_tile(bwi * fw, 128.f);
        float *decay_cache = reinterpret_cast<float *>(w_tiles + stage * kWarps);
        // Keep decay in this warp's own input tile; compacting across
        // warps here would race with unfinished reads of raw w.
        publish_decay(fw, decay_cache + warp * (128 - 16), warp);
        SharedTile *cached_ut = backward_cache;
        SharedTile *cached_du = cached_ut + kWarps;
        SharedTile *cached_ds = cached_du + kWarps;
        for (int i = 0; i < kWarps; ++i)
            cached_ds[warp * kWarps + i] = HalfTile(dstate[i]);
        __syncthreads();
        FloatTile ab = tril<1>(wa % bwi), ak_float = tril<1>(wa % kwi);
        reduce_pair(ab, ak_float, (float *)scratch, false);
        HalfTile ak = ak_float;

        FloatTile ab_inv = unit_lower_inverse(ab);

        HalfTile ut;
        {
            FloatTile ab_ut = vt % ak;
            for (int i = 0; i < kWarps; i++)
                ab_ut += state[warp * kWarps + i] % (HalfTile)a_tiles[stage * kWarps + i];
            ut = refined_product(ab_ut, ab_inv);
        }
        cached_ut[warp] = transpose(ut);

        // Backward does not consume y; omit its redundant reductions/barriers.
        FloatTile qb_float = tril<0>(wq % bwi), qk_float = tril<0>(wq % kwi);
        reduce_pair(qb_float, qk_float, (float *)scratch);
        HalfTile qb = qb_float, qk = qk_float;

        HalfTile dyt = sdy.t();
        FloatTile dut = FloatTile(dyt % transpose(qb));
        FloatTile dv = transpose(qk) % dyt;
        for (int i = 0; i < kWarps; i++) {
            HalfTile dstatei = dstate[i];
            dut += dstatei % (HalfTile)b_tiles[stage * kWarps + i];
            dv += (HalfTile)k_tiles[stage * kWarps + i] % dstatei;
        }
        HalfTile dab_ut = refined_product(dut, transpose_float(ab_inv));
        cached_du[warp] = transpose(dab_ut);
        dv += transpose(ak) % dab_ut;

        int off = (first_chunk + t) * K * H * C + head * C + warp * 16;
        int outoff = (batch*original_T+t*K)*H*C+head*C+warp*16;
        store_gradient(dv, dv_ + outoff, stride, inv_grad_scale, t*K, original_T, mask+batch*original_T);

        FloatTile dab = tril<1>(transpose(dab_ut) % transpose(ut)),
                  dak = tril<1>(transpose(dab_ut) % transpose(vt));
        reduce_pair(dab, dak, (float *)scratch);
        FloatTile dab_u_state0;
        dab_u_state0.zero_();
        for (int i = 0; i < kWarps; i++)
            dab_u_state0 += (HalfTile)cached_du[i] % state[i * kWarps + warp].t();

        FloatTile da = dab_u_state0;
        da += dab % transpose(bwi);
        da += dak % transpose(kwi);
        FloatTile chain_a = da * wa;
        da = non_incl_pref * da;
        store_gradient(da, da_ + outoff, stride, inv_grad_scale, t*K, original_T, mask+batch*original_T);

        FloatTile dqb = tril<0>(transpose(dyt) % transpose(ut)),
                  dqk = tril<0>(transpose(dyt) % transpose(vt));
        reduce_pair(dqb, dqk, (float *)scratch);
        FloatTile dy_state0;
        dy_state0.zero_();
        for (int i = 0; i < kWarps; i++)
            dy_state0 += (HalfTile)dy_tiles[stage * kWarps + i] % state[i * kWarps + warp].t();

        FloatTile dq = dy_state0;
        dq += dqb % transpose(bwi);
        dq += dqk % transpose(kwi);
        FloatTile chain = dq * wq;
        dq = incl_pref * dq;
        store_gradient(dq, dq_ + outoff, stride, inv_grad_scale, t*K, original_T, mask+batch*original_T);

        HalfTile wqt = transpose(wq), wat = transpose(wa);

        FloatTile u_dstate, v_dstate, dw;
        u_dstate.zero_();
        v_dstate.zero_();
        dw.zero_();
        for (int i = 0; i < kWarps; i++) {
            HalfTile dstatei = cached_ds[i * kWarps + warp];
            HalfTile dstatei_t = transpose(dstatei);
            v_dstate += (HalfTile)v_tiles[stage * kWarps + i] % dstatei_t;
            u_dstate += (HalfTile)cached_ut[i] % dstatei_t;
            dw += row_sum_broadcast((HalfTile)state[i * kWarps + warp].t() * dstatei_t);
        }

        FloatTile db = scale_tile(fw * u_dstate, 128.f);
        db += transpose(dab) % wat;
        db += transpose(dqb) % wqt;
        FloatTile chain_b = db * bwi;
        db = inv_incl_pref * db;
        store_gradient(db, db_ + outoff, stride, inv_grad_scale, t*K, original_T, mask+batch*original_T);

        FloatTile dk = scale_tile(fw * v_dstate, 128.f);
        dk += transpose(dak) % wat;
        dk += transpose(dqk) % wqt;
        FloatTile chain_k = dk * kwi;
        dk = inv_incl_pref * dk;
        store_gradient(dk, dk_ + outoff, stride, inv_grad_scale, t*K, original_T, mask+batch*original_T);

        dw = scale_tile(fw * dw, 128.f);
        FloatTile end_terms = v_dstate * scale_tile(fw * kwi, 128.f);
        end_terms += u_dstate * scale_tile(fw * bwi, 128.f);
        // cumsumv also leaves its input as the total, broadcast to every row.
        cumsumv<1, 0>(end_terms);
        dw += end_terms;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            chain.fdata[j] += chain_a.fdata[j] - chain_k.fdata[j] - chain_b.fdata[j];
        FloatTile suffix = cumsumv<1, 1>(chain);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            dw.fdata[j] += suffix.fdata[j] - chain_a.fdata[j];

        HalfTile raw_w;
        raw_w = GlobalBf16Tile(w_ + off, stride);
        FloatTile dw_fac = raw_w;
        apply_(dw_fac, [](float x) {
            float z = 1.f / (1.f + __expf(-x));
            return -0.6065306597126334f * z * (1.f - z);
        });
        dw = dw * dw_fac;
        store_gradient(dw, dw_ + outoff, stride, inv_grad_scale, t*K, original_T, mask+batch*original_T);

        for (int i = 0; i < kWarps; i++) {
            FloatTile ndstate = dstate[i] * cached_decay(decay_cache + i * (128 - 16), i);
            ndstate += dyt % (HalfTile)q_tiles[stage * kWarps + i].t();
            ndstate += dab_ut %
                       HalfTile(scale_tile((HalfTile)a_tiles[stage * kWarps + i].t(), 1.f / 128.f));
            dstate[i] = ndstate;
        }
    }
    for (int i=0;i<kWarps;++i) {
        for (int j=0;j<8;++j) {
            int value=warp*16+(threadIdx.x%32)/4+((j/2)%2)*8;
            int key=i*16+(threadIdx.x%4)*2+j%2+(j/4)*8;
            long long off=(long long)(batch*H+head)*C*C+key*C+value;
            initial_state_grad[off]=dstate[i].fdata[j]*inv_grad_scale;
        }
    }
}

// Adapter-owned packing and PyTorch binding. Forward remains rwkv-native FP32.
#include <math_constants.h>
struct PackedInputs {
    const float *x[7];
    __nv_bfloat16 *packed[7];
};

__global__ void pack_native(PackedInputs p,const float *cp,const bool *mask,
                            __half *out_cp,float *maxima,int T,int TP,int H) {
    int bh=blockIdx.x,c=blockIdx.y,bb=bh/H,head=bh%H,count=TP/16;
    __shared__ float state[64][65];
    __shared__ float partial[4];
    float largest=0.f;
    for(int z=threadIdx.x;z<16*64;z+=blockDim.x) {
        int t=c*16+z/64,j=z%64;
        bool valid=t<T && mask[bb*T+t];
        long long src=((long long)bb*T+t)*H*64+head*64+j;
        long long dst=((long long)bb*TP+t)*H*64+head*64+j;
        #pragma unroll
        for(int q=0;q<7;++q) {
            float v=valid?p.x[q][src]:(q==1?-CUDART_INF_F:0.f);
            p.packed[q][dst]=__float2bfloat16_rn(v);
            if(q==6)largest=fmaxf(largest,fabsf(v));
        }
    }
    long long base=((long long)bh*count+c)*4096;
    for(int z=threadIdx.x;z<4096;z+=blockDim.x)
        state[z/64][z%64]=cp[base+z]*(1.f/128.f);
    __syncthreads();
    for(int z=threadIdx.x;z<4096;z+=blockDim.x)
        out_cp[base+z]=__float2half_rn(state[z%64][z/64]);
    for(int offset=16;offset;offset>>=1)
        largest=fmaxf(largest,__shfl_xor_sync(0xffffffff,largest,offset));
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=largest;
    __syncthreads();
    if(threadIdx.x==0)
        maxima[bh*count+c]=fmaxf(fmaxf(partial[0],partial[1]),fmaxf(partial[2],partial[3]));
}

__global__ void reduce_ranges(const float *maxima,const float *ds,float *ranges,int count) {
    int bh=blockIdx.x;
    float largest=0.f;
    for(int j=threadIdx.x;j<count;j+=blockDim.x)largest=fmaxf(largest,maxima[bh*count+j]);
    if(ds)for(int j=threadIdx.x;j<4096;j+=blockDim.x)largest=fmaxf(largest,fabsf(ds[(long long)bh*4096+j]));
    for(int offset=16;offset;offset>>=1)
        largest=fmaxf(largest,__shfl_xor_sync(0xffffffff,largest,offset));
    __shared__ float partial[4];
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=largest;
    __syncthreads();
    if(threadIdx.x==0)
        ranges[bh]=fmaxf(fmaxf(partial[0],partial[1]),fmaxf(partial[2],partial[3]));
}

std::vector<at::Tensor> run(std::vector<at::Tensor> inputs,at::Tensor mask,
                           at::Tensor checkpoints,at::Tensor dy,at::Tensor ds) {
    TORCH_CHECK(inputs.size()==6,"six WKV operands required");
    auto r=inputs[0];
    TORCH_CHECK(r.is_cuda() && r.scalar_type()==at::kFloat && r.dim()==4 &&
                r.size(0)>0 && r.size(1)>0 && r.size(2)>0 && r.size(3)==64,"FP32 [B,T,H,64]");
    c10::cuda::CUDAGuard guard(r.device());
    for(auto x:inputs)TORCH_CHECK(x.device()==r.device() && x.scalar_type()==at::kFloat &&
                                  x.sizes()==r.sizes() && x.is_contiguous(),"operand layout");
    int B=r.size(0),T=r.size(1),H=r.size(2),TP=(T+15)/16*16,count=TP/16;
    TORCH_CHECK(dy.device()==r.device() && dy.scalar_type()==at::kFloat && dy.is_contiguous() &&
                dy.sizes()==r.sizes(),"FP32 dy");
    TORCH_CHECK(mask.device()==r.device() && mask.scalar_type()==at::kBool && mask.is_contiguous() &&
                mask.sizes()==at::IntArrayRef({B,T}),"bool [B,T] mask");
    TORCH_CHECK(checkpoints.device()==r.device() && checkpoints.scalar_type()==at::kFloat &&
                checkpoints.is_contiguous() && checkpoints.sizes()==at::IntArrayRef({B,H,count,64,64}),
                "native FP32 checkpoint interval must be 16");
    TORCH_CHECK(ds.device()==r.device() && ds.scalar_type()==at::kFloat && ds.is_contiguous() &&
                (ds.numel()==0 || ds.sizes()==at::IntArrayRef({B,H,64,64})),"FP32 final-state adjoint");
    auto packed=at::empty({7,B,TP,H,64},r.options().dtype(at::kBFloat16));
    auto cp=at::empty({B,H,count,64,64},r.options().dtype(at::kHalf));
    auto maxima=at::empty({B*H,count},r.options()),ranges=at::empty({B*H},r.options());
    auto grads=at::empty({6,B,T,H,64},r.options());
    auto initial_grad=at::empty({B,H,64,64},r.options());
    PackedInputs p{};
    long long packed_stride=(long long)B*TP*H*64,grad_stride=r.numel();
    for(int q=0;q<7;++q) {
        p.x[q]=(q==6?dy:inputs[q]).data_ptr<float>();
        p.packed[q]=reinterpret_cast<__nv_bfloat16*>(packed.data_ptr())+q*packed_stride;
    }
    auto stream=c10::cuda::getCurrentCUDAStream();
    const float *terminal=ds.numel()?ds.data_ptr<float>():nullptr;
    pack_native<<<dim3(B*H,count),128,0,stream>>>(p,checkpoints.data_ptr<float>(),mask.data_ptr<bool>(),
               reinterpret_cast<__half*>(cp.data_ptr()),maxima.data_ptr<float>(),T,TP,H);
    reduce_ranges<<<B*H,128,0,stream>>>(maxima.data_ptr<float>(),terminal,ranges.data_ptr<float>(),count);
    constexpr int shared_mem=sizeof(SharedTile)*(kBackwardStages*kWarps*7+24)+sizeof(float)*32*8*kWarps*2;
    C10_CUDA_CHECK(cudaFuncSetAttribute(backward_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_mem));
    float *g=grads.data_ptr<float>();
    backward_kernel<<<dim3(H,B),128,shared_mem,stream>>>(TP,H,
        p.packed[1],p.packed[0],p.packed[2],p.packed[3],p.packed[4],p.packed[5],p.packed[6],
        reinterpret_cast<float*>(cp.data_ptr()),terminal,
        g+grad_stride,g,g+2*grad_stride,g+3*grad_stride,g+4*grad_stride,g+5*grad_stride,
        initial_grad.data_ptr<float>(),ranges.data_ptr<float>(),mask.data_ptr<bool>(),T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grads[0],grads[1],grads[2],grads[3],grads[4],grads[5],initial_grad};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}

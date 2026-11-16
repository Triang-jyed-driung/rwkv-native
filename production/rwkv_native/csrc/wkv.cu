// Independent implementation; external state and checkpoints use [B,H,K,V].
// For each value row, S' = S diag(d) + (S a) b^T + v k^T.
// Each thread owns one value row; all state arithmetic and adjoints are FP32.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

using at::Tensor;
constexpr int D = 64;
constexpr float DECAY_SCALE = -0.6065306597126334f;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return __shfl_sync(0xffffffff, x, 0);
}

template <typename scalar_t>
__global__ void prepare_decay(const scalar_t *w, float *decay, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        decay[i] = expf(DECAY_SCALE / (1.f + expf(-float(w[i]))));
}

// A warp owns value rows, two keys per lane. Key reductions preserve the
// identical pair-product tree used by the thread-per-value tile kernel.
template <typename scalar_t, bool CHECKPOINT, int ROWS>
__global__ void forward_warp_kernel(const scalar_t *__restrict__ r, const scalar_t *__restrict__ w,
                                    const scalar_t *__restrict__ k, const scalar_t *__restrict__ v,
                                    const scalar_t *__restrict__ a, const scalar_t *__restrict__ b,
                                    const float *__restrict__ initial,
                                    const bool *__restrict__ mask, scalar_t *__restrict__ y,
                                    float *__restrict__ final, float *__restrict__ checkpoints,
                                    const float *__restrict__ decay, int T, int H, int chunk,
                                    int count) {
    int lane = threadIdx.x % 32, row = (blockIdx.y * 4 + threadIdx.x / 32) * ROWS;
    int batch = blockIdx.x / H, head = blockIdx.x % H;
    int64_t base = int64_t(blockIdx.x) * D * D;
    float s0[ROWS], s1[ROWS];
#pragma unroll
    for (int q = 0; q < ROWS; ++q) {
        s0[q] = initial[base + lane * D + row + q];
        s1[q] = initial[base + (lane + 32) * D + row + q];
    }
    for (int t = 0; t < T; ++t) {
        if constexpr (CHECKPOINT) {
            if (t % chunk == 0) {
                int64_t cp = (int64_t(blockIdx.x) * count + t / chunk) * D * D;
#pragma unroll
                for (int q = 0; q < ROWS; ++q) {
                    checkpoints[cp + lane * D + row + q] = s0[q];
                    checkpoints[cp + (lane + 32) * D + row + q] = s1[q];
                }
            }
        }
        int64_t idx = ((int64_t(batch) * T + t) * H + head) * D;
        if (!mask[int64_t(batch) * T + t]) {
            if (lane == 0) {
#pragma unroll
                for (int q = 0; q < ROWS; ++q)
                    y[idx + row + q] = scalar_t(0.f);
            }
            continue;
        }
        float d0 =
            decay ? decay[idx + lane] : expf(DECAY_SCALE / (1.f + expf(-float(w[idx + lane]))));
        float d1 = decay ? decay[idx + lane + 32]
                         : expf(DECAY_SCALE / (1.f + expf(-float(w[idx + lane + 32]))));
        float a0 = float(a[idx + lane]), a1 = float(a[idx + lane + 32]);
        float b0 = float(b[idx + lane]), b1 = float(b[idx + lane + 32]);
        float k0 = float(k[idx + lane]), k1 = float(k[idx + lane + 32]);
        float r0 = float(r[idx + lane]), r1 = float(r[idx + lane + 32]);
#pragma unroll
        for (int q = 0; q < ROWS; ++q) {
            float sa = warp_sum(fmaf(s0[q], a0, s1[q] * a1));
            float val = float(v[idx + row + q]);
            s0[q] = fmaf(s0[q], d0, fmaf(sa, b0, k0 * val));
            s1[q] = fmaf(s1[q], d1, fmaf(sa, b1, k1 * val));
            float out = warp_sum(fmaf(s0[q], r0, s1[q] * r1));
            if (lane == 0)
                y[idx + row + q] = scalar_t(out);
        }
    }
#pragma unroll
    for (int q = 0; q < ROWS; ++q) {
        final[base + lane * D + row + q] = s0[q];
        final[base + (lane + 32) * D + row + q] = s1[q];
    }
}

// For larger prefill batches, share token vectors across 64 value rows.
// Unlike warp-per-row execution, each CTA fetches each token vector only once.
template <typename scalar_t, bool CHECKPOINT, bool INLINE_DECAY>
__global__ void forward_tile_kernel(const scalar_t *__restrict__ r, const scalar_t *__restrict__ w,
                                    const float *__restrict__ decay, const scalar_t *__restrict__ k,
                                    const scalar_t *__restrict__ v, const scalar_t *__restrict__ a,
                                    const scalar_t *__restrict__ b,
                                    const float *__restrict__ initial,
                                    const bool *__restrict__ mask, scalar_t *__restrict__ y,
                                    float *__restrict__ final, float *__restrict__ checkpoints,
                                    int T, int H, int chunk, int count) {
    int lane = threadIdx.x, row = lane;
    int batch = blockIdx.x / H, head = blockIdx.x % H;
    int64_t base = int64_t(blockIdx.x) * D * D;
    float s[D];
    __shared__ float rr[D], dd[D], kk[D], aa[D], bb[D];
#pragma unroll
    for (int j = 0; j < D; ++j)
        s[j] = initial[base + j * D + row];
    for (int t = 0; t < T; ++t) {
        if constexpr (CHECKPOINT) {
            if (t % chunk == 0) {
                int64_t cp = (int64_t(blockIdx.x) * count + t / chunk) * D * D + row * D;
#pragma unroll
                for (int j = 0; j < D; ++j)
                    checkpoints[cp - row * D + j * D + row] = s[j];
            }
        }
        int64_t idx = ((int64_t(batch) * T + t) * H + head) * D;
        if (!mask[int64_t(batch) * T + t]) {
            y[idx + row] = scalar_t(0.f);
            continue;
        }
        __syncthreads();
#pragma unroll
        for (int j = lane; j < D; j += 64) {
            rr[j] = float(r[idx + j]);
            if constexpr (INLINE_DECAY)
                dd[j] = expf(DECAY_SCALE / (1.f + expf(-float(w[idx + j]))));
            else
                dd[j] = decay[idx + j];
            kk[j] = float(k[idx + j]);
            aa[j] = float(a[idx + j]);
            bb[j] = float(b[idx + j]);
        }
        __syncthreads();
        float parts[32], val = float(v[idx + row]);
#pragma unroll
        for (int j = 0; j < 32; ++j)
            parts[j] = fmaf(s[j], aa[j], s[j + 32] * aa[j + 32]);
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
            for (int j = 0; j < offset; ++j)
                parts[j] += parts[j + offset];
        }
        float sa = parts[0];
#pragma unroll
        for (int j = 0; j < D; ++j)
            s[j] = fmaf(s[j], dd[j], fmaf(sa, bb[j], kk[j] * val));
#pragma unroll
        for (int j = 0; j < 32; ++j)
            parts[j] = fmaf(s[j], rr[j], s[j + 32] * rr[j + 32]);
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
            for (int j = 0; j < offset; ++j)
                parts[j] += parts[j + offset];
        }
        y[idx + row] = scalar_t(parts[0]);
    }
#pragma unroll
    for (int j = 0; j < D; ++j)
        final[base + j * D + row] = s[j];
}

// Eight lanes cooperate on each value row. Checkpoints are read directly in
// their strided layout; replay writes coalesced states without a staging copy.
__device__ __forceinline__ float sum8(float x) {
#pragma unroll
    for (int offset = 4; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset, 8);
    return x;
}

template <typename scalar_t>
__global__ void replay_parallel(const float *decay, const scalar_t *k, const scalar_t *v,
                                const scalar_t *a, const scalar_t *b, const float *cp,
                                const bool *mask, float *replay, int T, int H, int chunk, int count,
                                int c) {
    int lane = threadIdx.x % 8, row = blockIdx.y * 16 + threadIdx.x / 8;
    int batch = blockIdx.x / H, head = blockIdx.x % H;
    int64_t base = (int64_t(blockIdx.x) * count + c) * D * D;
    float s[8];
#pragma unroll
    for (int q = 0; q < 8; ++q)
        s[q] = cp[base + (lane + 8 * q) * D + row];
    for (int t = c * chunk; t < min(T, (c + 1) * chunk); ++t) {
        int64_t rp = (int64_t(blockIdx.x) * chunk + t - c * chunk) * D * D + row * D;
#pragma unroll
        for (int q = 0; q < 8; ++q)
            replay[rp + lane + 8 * q] = s[q];
        if (!mask[int64_t(batch) * T + t])
            continue;
        int64_t idx = ((int64_t(batch) * T + t) * H + head) * D;
        float sa = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q)
            sa = fmaf(s[q], float(a[idx + lane + 8 * q]), sa);
        sa = sum8(sa);
        float val = float(v[idx + row]);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            int j = lane + 8 * q;
            s[q] = fmaf(s[q], decay[idx + j], fmaf(sa, float(b[idx + j]), float(k[idx + j]) * val));
        }
    }
}

template <typename scalar_t>
__global__ void backward_parallel(const scalar_t *r, const scalar_t *w, const scalar_t *k,
                                  const scalar_t *v, const scalar_t *a, const scalar_t *b,
                                  const bool *mask, const float *replay, const scalar_t *dy,
                                  float *ds, scalar_t *dr, scalar_t *dw, scalar_t *dk, scalar_t *dv,
                                  scalar_t *da, scalar_t *db, int T, int H, int chunk, int start,
                                  int stop, const float *incoming_gradient) {
    int tid = threadIdx.x, lane = tid % 8, row = tid / 8;
    int key = tid % 64, part = tid / 64;
    int batch = blockIdx.x / H, head = blockIdx.x % H;
    int64_t base = int64_t(blockIdx.x) * D * D;
    float grad[8];
    // XOR row swizzle makes both 8-lane row access and column access bank-conflict free.
    __shared__ float old[D][D], g[D][D];
    __shared__ float rr[D], dd[D], kk[D], vv[D], aa[D], bb[D], yy[D], sa[D], ga[D];
    __shared__ float partial[5][8][D];
#pragma unroll
    for (int q = 0; q < 8; ++q)
        grad[q] = incoming_gradient ? incoming_gradient[base + (lane + q * 8) * D + row] : 0.f;
    for (int t = stop - 1; t >= start; --t) {
        int64_t idx = ((int64_t(batch) * T + t) * H + head) * D;
        if (!mask[int64_t(batch) * T + t]) {
            if (tid < D)
                dr[idx + tid] = dw[idx + tid] = dk[idx + tid] = dv[idx + tid] = da[idx + tid] =
                    db[idx + tid] = scalar_t(0.f);
            continue;
        }
        __syncthreads();
        if (tid < D) {
            rr[tid] = float(r[idx + tid]);
            dd[tid] = expf(DECAY_SCALE / (1.f + expf(-float(w[idx + tid]))));
            kk[tid] = float(k[idx + tid]);
            vv[tid] = float(v[idx + tid]);
            aa[tid] = float(a[idx + tid]);
            bb[tid] = float(b[idx + tid]);
            yy[tid] = float(dy[idx + tid]);
        }
        int64_t rp = (int64_t(blockIdx.x) * chunk + t - start) * D * D + row * D;
#pragma unroll
        for (int q = 0; q < 8; ++q)
            old[row][(lane + q * 8) ^ ((row & 3) * 8)] = replay[rp + lane + q * 8];
        __syncthreads();
        float sai = 0.f, gai = 0.f, dvi = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            int j = lane + q * 8;
            sai = fmaf(old[row][j ^ ((row & 3) * 8)], aa[j], sai);
            grad[q] = fmaf(yy[row], rr[j], grad[q]);
            g[row][j ^ ((row & 3) * 8)] = grad[q];
            gai = fmaf(grad[q], bb[j], gai);
            dvi = fmaf(grad[q], kk[j], dvi);
        }
        sai = sum8(sai);
        gai = sum8(gai);
        dvi = sum8(dvi);
        if (lane == 0) {
            sa[row] = sai;
            ga[row] = gai;
            dv[idx + row] = scalar_t(dvi);
        }
        __syncthreads();
        float dri = 0.f, dwi = 0.f, dki = 0.f, dai = 0.f, dbi = 0.f;
#pragma unroll
        for (int i = part; i < D; i += 8) {
            float ns =
                fmaf(old[i][key ^ ((i & 3) * 8)], dd[key], fmaf(sa[i], bb[key], kk[key] * vv[i]));
            dri = fmaf(ns, yy[i], dri);
            dwi = fmaf(g[i][key ^ ((i & 3) * 8)], old[i][key ^ ((i & 3) * 8)], dwi);
            dki = fmaf(g[i][key ^ ((i & 3) * 8)], vv[i], dki);
            dai = fmaf(ga[i], old[i][key ^ ((i & 3) * 8)], dai);
            dbi = fmaf(g[i][key ^ ((i & 3) * 8)], sa[i], dbi);
        }
        partial[0][part][key] = dri;
        partial[1][part][key] = dwi;
        partial[2][part][key] = dki;
        partial[3][part][key] = dai;
        partial[4][part][key] = dbi;
#pragma unroll
        for (int q = 0; q < 8; ++q)
            grad[q] = fmaf(grad[q], dd[lane + q * 8], gai * aa[lane + q * 8]);
        __syncthreads();
        if (tid < D) {
            float sums[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int p = 0; p < 8; ++p) {
#pragma unroll
                for (int n = 0; n < 5; ++n)
                    sums[n] += partial[n][p][tid];
            }
            float sig = 1.f / (1.f + expf(-float(w[idx + tid])));
            dr[idx + tid] = scalar_t(sums[0]);
            dw[idx + tid] = scalar_t(sums[1] * dd[tid] * DECAY_SCALE * sig * (1.f - sig));
            dk[idx + tid] = scalar_t(sums[2]);
            da[idx + tid] = scalar_t(sums[3]);
            db[idx + tid] = scalar_t(sums[4]);
        }
    }
#pragma unroll
    for (int q = 0; q < 8; ++q)
        ds[base + (lane + q * 8) * D + row] = grad[q];
}

void check_inputs(const Tensor &r, const std::vector<Tensor> &xs, const Tensor &mask) {
    TORCH_CHECK(r.is_cuda() && r.is_contiguous() && r.dim() == 4 && r.size(3) == D && r.numel() > 0,
                "r must be nonempty contiguous CUDA [B,T,H,64]");
    TORCH_CHECK(r.scalar_type() == at::kFloat || r.scalar_type() == at::kHalf ||
                    r.scalar_type() == at::kBFloat16,
                "unsupported input dtype");
    for (const auto &x : xs)
        TORCH_CHECK(x.device() == r.device() && x.sizes() == r.sizes() &&
                        x.scalar_type() == r.scalar_type() && x.is_contiguous(),
                    "all vectors must match r");
    TORCH_CHECK(mask.device() == r.device() && mask.scalar_type() == at::kBool &&
                    mask.is_contiguous() && mask.dim() == 2 && mask.size(0) == r.size(0) &&
                    mask.size(1) == r.size(1),
                "invalid mask");
}

std::vector<Tensor> wkv_forward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b,
                                Tensor initial, Tensor mask, int64_t chunk, bool save) {
    check_inputs(r, {w, k, v, a, b}, mask);
    TORCH_CHECK(chunk >= 1 && chunk <= 256, "invalid checkpoint size");
    c10::cuda::CUDAGuard guard(r.device());
    int B = r.size(0), T = r.size(1), H = r.size(2), count = (T + chunk - 1) / chunk;
    TORCH_CHECK(initial.device() == r.device() && initial.scalar_type() == at::kFloat &&
                    initial.is_contiguous() && initial.sizes() == at::IntArrayRef({B, H, D, D}),
                "invalid initial state");
    auto y = at::empty_like(r), final = at::empty_like(initial);
    auto cp = at::empty({B, H, save ? count : 0, D, D}, initial.options());
    auto decay = at::empty({T > 1 ? r.numel() : 0}, initial.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, r.scalar_type(), "wkv_forward", [&] {
        if (T > 1)
            prepare_decay<scalar_t><<<(r.numel() + 255) / 256, 256, 0, stream>>>(
                w.data_ptr<scalar_t>(), decay.data_ptr<float>(), r.numel());
        const float *decay_ptr = T > 1 ? decay.data_ptr<float>() : nullptr;
        if ((B * H >= 64 && T >= 8) || (T == 1 && B * H >= 512)) {
            if (save) {
                if (T == 1)
                    forward_tile_kernel<scalar_t, true, true><<<dim3(B * H, 1), 64, 0, stream>>>(
                        r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), decay_ptr,
                        k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                        b.data_ptr<scalar_t>(), initial.data_ptr<float>(), mask.data_ptr<bool>(),
                        y.data_ptr<scalar_t>(), final.data_ptr<float>(), cp.data_ptr<float>(), T, H,
                        chunk, count);
                else
                    forward_tile_kernel<scalar_t, true, false><<<dim3(B * H, 1), 64, 0, stream>>>(
                        r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), decay_ptr,
                        k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                        b.data_ptr<scalar_t>(), initial.data_ptr<float>(), mask.data_ptr<bool>(),
                        y.data_ptr<scalar_t>(), final.data_ptr<float>(), cp.data_ptr<float>(), T, H,
                        chunk, count);
            } else {
                if (T == 1)
                    forward_tile_kernel<scalar_t, false, true><<<dim3(B * H, 1), 64, 0, stream>>>(
                        r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), decay_ptr,
                        k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                        b.data_ptr<scalar_t>(), initial.data_ptr<float>(), mask.data_ptr<bool>(),
                        y.data_ptr<scalar_t>(), final.data_ptr<float>(), nullptr, T, H, chunk,
                        count);
                else
                    forward_tile_kernel<scalar_t, false, false><<<dim3(B * H, 1), 64, 0, stream>>>(
                        r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), decay_ptr,
                        k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(),
                        b.data_ptr<scalar_t>(), initial.data_ptr<float>(), mask.data_ptr<bool>(),
                        y.data_ptr<scalar_t>(), final.data_ptr<float>(), nullptr, T, H, chunk,
                        count);
            }
        } else if (T > 1) {
            if (save)
                forward_warp_kernel<scalar_t, true, 1><<<dim3(B * H, 16), 128, 0, stream>>>(
                    r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                    initial.data_ptr<float>(), mask.data_ptr<bool>(), y.data_ptr<scalar_t>(),
                    final.data_ptr<float>(), cp.data_ptr<float>(), decay_ptr, T, H, chunk, count);
            else
                forward_warp_kernel<scalar_t, false, 1><<<dim3(B * H, 16), 128, 0, stream>>>(
                    r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                    initial.data_ptr<float>(), mask.data_ptr<bool>(), y.data_ptr<scalar_t>(),
                    final.data_ptr<float>(), nullptr, decay_ptr, T, H, chunk, count);
        } else {
            if (save)
                forward_warp_kernel<scalar_t, true, 4><<<dim3(B * H, 4), 128, 0, stream>>>(
                    r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                    initial.data_ptr<float>(), mask.data_ptr<bool>(), y.data_ptr<scalar_t>(),
                    final.data_ptr<float>(), cp.data_ptr<float>(), nullptr, T, H, chunk, count);
            else
                forward_warp_kernel<scalar_t, false, 4><<<dim3(B * H, 4), 128, 0, stream>>>(
                    r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                    initial.data_ptr<float>(), mask.data_ptr<bool>(), y.data_ptr<scalar_t>(),
                    final.data_ptr<float>(), nullptr, nullptr, T, H, chunk, count);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, final, cp};
}

std::vector<Tensor> wkv_backward(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b,
                                 Tensor mask, Tensor cp, Tensor dy, Tensor final_grad,
                                 int64_t chunk) {
    check_inputs(r, {w, k, v, a, b, dy}, mask);
    TORCH_CHECK(chunk >= 1 && chunk <= 256, "invalid checkpoint size");
    c10::cuda::CUDAGuard guard(r.device());
    // With one checkpoint, compact the replay stride to the actual sequence.
    chunk = std::min(chunk, r.size(1));
    int B = r.size(0), T = r.size(1), H = r.size(2), count = (T + chunk - 1) / chunk;
    TORCH_CHECK(cp.device() == r.device() && cp.scalar_type() == at::kFloat && cp.is_contiguous() &&
                    cp.sizes() == at::IntArrayRef({B, H, count, D, D}),
                "invalid checkpoints");
    TORCH_CHECK(
        final_grad.device() == r.device() && final_grad.scalar_type() == at::kFloat &&
            final_grad.is_contiguous() &&
            (final_grad.numel() == 0 || final_grad.sizes() == at::IntArrayRef({B, H, D, D})),
        "invalid final state gradient");
    auto dr = at::empty_like(r), dw = at::empty_like(r), dk = at::empty_like(r);
    auto dv = at::empty_like(r), da = at::empty_like(r), db = at::empty_like(r);
    // Read the supplied adjoint once; an empty tensor means zero.
    auto ds = at::empty({B, H, D, D}, cp.options());
    auto replay = at::empty({B, H, chunk, D, D}, cp.options());
    auto decay = at::empty(r.sizes(), cp.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, r.scalar_type(), "wkv_backward", [&] {
        prepare_decay<scalar_t><<<(r.numel() + 255) / 256, 256, 0, stream>>>(
            w.data_ptr<scalar_t>(), decay.data_ptr<float>(), r.numel());
        for (int c = count - 1; c >= 0; --c) {
            int start = c * chunk, stop = std::min(T, int((c + 1) * chunk));
            replay_parallel<scalar_t><<<dim3(B * H, 4), 128, 0, stream>>>(
                decay.data_ptr<float>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), cp.data_ptr<float>(),
                mask.data_ptr<bool>(), replay.data_ptr<float>(), T, H, chunk, count, c);
            backward_parallel<scalar_t><<<B * H, 512, 0, stream>>>(
                r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(), replay.data_ptr<float>(), dy.data_ptr<scalar_t>(),
                ds.data_ptr<float>(), dr.data_ptr<scalar_t>(), dw.data_ptr<scalar_t>(),
                dk.data_ptr<scalar_t>(), dv.data_ptr<scalar_t>(), da.data_ptr<scalar_t>(),
                db.data_ptr<scalar_t>(), T, H, chunk, start, stop,
                c == count - 1 ? (final_grad.numel() ? final_grad.data_ptr<float>() : nullptr)
                               : ds.data_ptr<float>());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dr, dw, dk, dv, da, db, ds};
}

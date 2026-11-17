#pragma once
// Register prefetch for small head batches and long sequences.
// Load several independent token vectors before evaluating their recurrence.
// The pair-product reduction tree and increasing-token order are unchanged.
template <typename scalar_t, bool CHECKPOINT, int ROWS, int P>
__global__ void forward_register_kernel(const scalar_t *__restrict__ r, const scalar_t *__restrict__ w,
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
    for (int start = 0; start < T; start += P) {
        float dd0[P], dd1[P], aa0[P], aa1[P], bb0[P], bb1[P];
        float kk0[P], kk1[P], rr0[P], rr1[P], vv[P][ROWS];
        #pragma unroll
        for (int p = 0; p < P; ++p) {
            const int64_t idx = ((int64_t(batch)*T + start+p)*H + head)*64;
            if (start+p < T) {
                dd0[p]=decay[idx+lane]; dd1[p]=decay[idx+lane+32];
                aa0[p]=a[idx+lane]; aa1[p]=a[idx+lane+32];
                bb0[p]=b[idx+lane]; bb1[p]=b[idx+lane+32];
                kk0[p]=k[idx+lane]; kk1[p]=k[idx+lane+32];
                rr0[p]=r[idx+lane]; rr1[p]=r[idx+lane+32];
                #pragma unroll
                for(int q=0;q<ROWS;++q)vv[p][q]=v[idx+row+q];
            }
        }
        #pragma unroll
        for (int p = 0; p < P; ++p) {
        const int t = start+p;
        if(t>=T)continue;
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
        float d0=dd0[p],d1=dd1[p],a0=aa0[p],a1=aa1[p];
        float b0=bb0[p],b1=bb1[p],k0=kk0[p],k1=kk1[p],r0=rr0[p],r1=rr1[p];
#pragma unroll
        for (int q = 0; q < ROWS; ++q) {
            float sa = warp_sum(fmaf(s0[q], a0, s1[q] * a1));
            float val = vv[p][q];
            s0[q] = fmaf(s0[q], d0, fmaf(sa, b0, k0 * val));
            s1[q] = fmaf(s1[q], d1, fmaf(sa, b1, k1 * val));
            float out = warp_sum(fmaf(s0[q], r0, s1[q] * r1));
            if (lane == 0)
                y[idx + row + q] = scalar_t(out);
        }
        }
    }
#pragma unroll
    for (int q = 0; q < ROWS; ++q) {
        final[base + lane * D + row + q] = s0[q];
        final[base + (lane + 32) * D + row + q] = s1[q];
    }
}

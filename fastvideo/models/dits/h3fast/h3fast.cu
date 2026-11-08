// SPDX-License-Identifier: Apache-2.0
// Fused memory-bound kernels of the MiniMax-H3 DiT block (inference, single GPU).
//
// The residual stream stays in compact tile order (tile rows contiguous, no pad rows). FP8 activations carry one scale
// per row: GEMMs run tensor-wise with scale 1 and the consumer of each GEMM output multiplies its row by that scale, so
// per-token scaling costs neither a second pass nor a slower GEMM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;
using fp8s = __nv_fp8_storage_t;

constexpr float kFp8Max = 448.f;
constexpr float kMinScale = 1.f / (448.f * 512.f);
constexpr float kFp4Range = 448.f * 6.f;  // a row's largest value lands at the top of block scale x E2M1
constexpr float kMinScale4 = 1e-12f;

__device__ __forceinline__ void ld8(const bf16* p, float (&f)[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ void ld4(const bf16* p, float (&f)[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ void st8(bf16* p, const float (&f)[8]) {
  __nv_bfloat162 h[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(h);
}

__device__ __forceinline__ void st4(bf16* p, const float (&f)[4]) {
  __nv_bfloat162 h[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint2*>(p) = *reinterpret_cast<const uint2*>(h);
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

__device__ __forceinline__ void st_fp8x8(fp8s* p, const float (&f)[8], float inv) {
  uint2 o;
  o.x = fp8x4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
  o.y = fp8x4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
  *reinterpret_cast<uint2*>(p) = o;
}

__device__ __forceinline__ uint32_t e2m1x4(float a, float b, float c, float d) {
  uint32_t r;
  asm("{\n\t.reg .b8 t0, t1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 t0, %2, %1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 t1, %4, %3;\n\t"
      "mov.b32 %0, {t0, t1, t0, t1};\n\t}"
      : "=r"(r)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return r & 0xffffu;
}

__device__ __forceinline__ float e4m3_value(uint32_t bits) {
  const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)bits, __NV_E4M3);
  return __half2float(__half(h));
}

// layout_128x4: 512-byte atoms of 128 rows x 4 scale columns.
__device__ __forceinline__ size_t sf_offset(int64_t row, int col, int sf_cols) {
  return (size_t)(row >> 7) * (size_t)(sf_cols >> 2) * 512 + (size_t)(col >> 2) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (col & 3);
}

// One 16-element block's scale; the row's global scale is gs, so the block scale stays in E4M3 range.
__device__ __forceinline__ float fp4_block_scale(float vmax, float gs, uint32_t* bits) {
  *bits = __nv_cvt_float_to_fp8(gs * (vmax * (1.f / 6.f)), __NV_SATFINITE, __NV_E4M3);
  const float sf = e4m3_value(*bits);
  return (vmax != 0.f && sf != 0.f) ? gs / sf : 0.f;
}

// Eight elements per thread, a 16-block across two adjacent lanes. Element e of the row is t * 8 + i.
__device__ __forceinline__ void st_fp4x8(uint8_t* q, uint8_t* sfb, const float (&f)[8], float gs, int64_t row, int t,
                                         int sf_cols) {
  float vmax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) vmax = fmaxf(vmax, fabsf(f[i]));
  vmax = fmaxf(vmax, __shfl_xor_sync(~0u, vmax, 1));
  uint32_t bits;
  const float sc = fp4_block_scale(vmax, gs, &bits);
  const uint32_t lo = e2m1x4(f[0] * sc, f[1] * sc, f[2] * sc, f[3] * sc);
  const uint32_t hi = e2m1x4(f[4] * sc, f[5] * sc, f[6] * sc, f[7] * sc);
  *reinterpret_cast<uint32_t*>(q + t * 4) = lo | (hi << 16);
  if ((t & 1) == 0) sfb[sf_offset(row, t >> 1, sf_cols)] = (uint8_t)bits;
}

// Sixteen elements per thread: one whole block.
__device__ __forceinline__ void st_fp4x16(uint8_t* q, uint8_t* sfb, const float (&f)[2][8], float gs, int64_t row, int t,
                                          int sf_cols) {
  float vmax = 0.f;
#pragma unroll
  for (int c = 0; c < 2; ++c)
#pragma unroll
    for (int i = 0; i < 8; ++i) vmax = fmaxf(vmax, fabsf(f[c][i]));
  uint32_t bits;
  const float sc = fp4_block_scale(vmax, gs, &bits);
  uint2 o;
  o.x = e2m1x4(f[0][0] * sc, f[0][1] * sc, f[0][2] * sc, f[0][3] * sc) |
        (e2m1x4(f[0][4] * sc, f[0][5] * sc, f[0][6] * sc, f[0][7] * sc) << 16);
  o.y = e2m1x4(f[1][0] * sc, f[1][1] * sc, f[1][2] * sc, f[1][3] * sc) |
        (e2m1x4(f[1][4] * sc, f[1][5] * sc, f[1][6] * sc, f[1][7] * sc) << 16);
  *reinterpret_cast<uint2*>(q + t * 8) = o;
  sfb[sf_offset(row, t, sf_cols)] = (uint8_t)bits;
}

template <int NW, bool MAX>
__device__ __forceinline__ float block_reduce(float v, float* sm) {
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float u = __shfl_xor_sync(~0u, v, o);
    v = MAX ? fmaxf(v, u) : v + u;
  }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) sm[w] = v;
  __syncthreads();
  float r = l < NW ? sm[l] : 0.f;
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float u = __shfl_xor_sync(~0u, r, o);
    r = MAX ? fmaxf(r, u) : r + u;
  }
  __syncthreads();
  return r;
}

// x' = x + gate[idx] * (sy * y); then, with NORM, h = fp8(rmsnorm(x') * w * (1 + scale[idx]) + shift[idx]) with its row scale.
template <int NW, int R, bool MAX>
__device__ __forceinline__ void block_reduce_r(float (&v)[R], float* sm) {
#pragma unroll
  for (int k = 0; k < R; ++k)
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float u = __shfl_xor_sync(~0u, v[k], o);
      v[k] = MAX ? fmaxf(v[k], u) : v[k] + u;
    }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0)
#pragma unroll
    for (int k = 0; k < R; ++k) sm[k * 32 + w] = v[k];
  __syncthreads();
#pragma unroll
  for (int k = 0; k < R; ++k) {
    float r = l < NW ? sm[k * 32 + l] : 0.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float u = __shfl_xor_sync(~0u, r, o);
      r = MAX ? fmaxf(r, u) : r + u;
    }
    v[k] = r;
  }
  __syncthreads();
}

// R rows per CTA: all loads first, one pair of block reductions for the R rows (latency, not bandwidth, bound at R=1).
template <int C, bool BRANCH, bool NORM, bool FP4, int R>
__global__ void __launch_bounds__(C / 8) resid_norm_quant(const bf16* __restrict__ x, const bf16* __restrict__ y,
                                                          const float* __restrict__ sy, const bf16* __restrict__ gate,
                                                          const int* __restrict__ idx, const bf16* __restrict__ nw,
                                                          const bf16* __restrict__ shift, const bf16* __restrict__ scale,
                                                          int64_t gate_ld, int64_t tab_ld, bf16* __restrict__ xo,
                                                          fp8s* __restrict__ h, uint8_t* __restrict__ hsf,
                                                          float* __restrict__ hs, float eps, int64_t rows) {
  constexpr int NW = C / 8 / 32;
  __shared__ float sm[32 * R];
  const int t = threadIdx.x;
  const int64_t r0 = (int64_t)blockIdx.x * R;
  int64_t rr[R];
  int ti[R];
  float v[R][8];
#pragma unroll
  for (int k = 0; k < R; ++k) {
    rr[k] = min(r0 + k, rows - 1);  // a short last CTA repeats its last row; only real rows are stored
    ti[k] = idx[rr[k]];
    ld8(x + rr[k] * C + t * 8, v[k]);
  }
  if constexpr (BRANCH) {
#pragma unroll
    for (int k = 0; k < R; ++k) {
      float b[8], g[8];
      ld8(y + rr[k] * C + t * 8, b);
      ld8(gate + ti[k] * gate_ld + t * 8, g);
      const float s = sy[rr[k]];
#pragma unroll
      for (int i = 0; i < 8; ++i) v[k][i] = round_bf16(v[k][i] + g[i] * (s * b[i]));
    }
#pragma unroll
    for (int k = 0; k < R; ++k)
      if (r0 + k < rows) st8(xo + rr[k] * C + t * 8, v[k]);
  }
  if constexpr (NORM) {
    float wv[8];
    ld8(nw + t * 8, wv);
    float ss[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
      ss[k] = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss[k] += v[k][i] * v[k][i];
    }
    block_reduce_r<NW, R, false>(ss, sm);
    float am[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
      float sc[8], sf[8];
      ld8(scale + ti[k] * tab_ld + t * 8, sc);
      ld8(shift + ti[k] * tab_ld + t * 8, sf);
      const float rs = rsqrtf(ss[k] / C + eps);
      am[k] = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[k][i] = v[k][i] * rs * wv[i] * (1.f + sc[i]) + sf[i];
        am[k] = fmaxf(am[k], fabsf(v[k][i]));
      }
    }
    block_reduce_r<NW, R, true>(am, sm);
#pragma unroll
    for (int k = 0; k < R; ++k) {
      if (r0 + k >= rows) continue;
      const int64_t r = rr[k];
      if constexpr (FP4) {
        const float s = fmaxf(am[k] / kFp4Range, kMinScale4);
        st_fp4x8(reinterpret_cast<uint8_t*>(h) + r * (C / 2), hsf, v[k], 1.f / s, r, t, C / 16);
        if (t == 0) hs[r] = s;
      } else {
        const float s = fmaxf(am[k] / kFp8Max, kMinScale);
        st_fp8x8(h + r * C + t * 8, v[k], 1.f / s);
        if (t == 0) hs[r] = s;
      }
    }
  }
}

// One CTA per (tile, head): per-head q/k RMSNorm, partial RoPE, row scale, heads-first padded output, tile means.
// CC: also write q/k/v to compact BSHD copies (qc/kc/vc [rows, heads, D]) for the dense prefix passes on FA4.
template <int NWARP, bool CC>
__global__ void __launch_bounds__(NWARP * 32) qkv_prologue(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, int64_t ld,
    const float* __restrict__ s1, const int* __restrict__ tile_row0, const int* __restrict__ vbs,
    const bf16* __restrict__ wq, const bf16* __restrict__ wk, const bf16* __restrict__ cosb,
    const bf16* __restrict__ sinb, bf16* __restrict__ qb, bf16* __restrict__ kb, bf16* __restrict__ vb,
    float* __restrict__ qp, float* __restrict__ kp, float* __restrict__ vp, int heads, int64_t rows_pad, int n_tiles,
    int tile, float eps, bf16* __restrict__ qc, bf16* __restrict__ kc, bf16* __restrict__ vc) {
  constexpr int D = 128, ROT = 96, HALF_LANES = 12;
  __shared__ float sp[3][NWARP][D];
  const int h = blockIdx.x % heads, ti = blockIdx.x / heads;
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  const int row0 = tile_row0[ti], n = vbs[ti];
  const int partner = l < HALF_LANES ? l + HALF_LANES : (l < 2 * HALF_LANES ? l - HALF_LANES : l);
  const float sign = l < HALF_LANES ? -1.f : 1.f;
  const bool rot = l < 2 * HALF_LANES;
  float wqv[4], wkv[4];
  ld4(wq + l * 4, wqv);
  ld4(wk + l * 4, wkv);
  float pq[4] = {0.f, 0.f, 0.f, 0.f}, pk[4] = {0.f, 0.f, 0.f, 0.f}, pv[4] = {0.f, 0.f, 0.f, 0.f};
  const size_t plane = (size_t)h * rows_pad + (size_t)ti * tile;
  // Two rows in flight per warp (i and i + NWARP): both rows' loads issue before either row's math.
  auto row = [&](int i, float (&qf)[4], float (&kf)[4], float (&vf)[4], float s, const float (&c)[4],
                 const float (&sn)[4]) {
    float sq = 0.f, sk = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qf[j] *= s;
      kf[j] *= s;
      vf[j] *= s;
      sq += qf[j] * qf[j];
      sk += kf[j] * kf[j];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      sq += __shfl_xor_sync(~0u, sq, o);
      sk += __shfl_xor_sync(~0u, sk, o);
    }
    const float rq = rsqrtf(sq / D + eps), rk = rsqrtf(sk / D + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qf[j] *= rq * wqv[j];
      kf[j] *= rk * wkv[j];
    }
    float qo[4], ko[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float qr = __shfl_sync(~0u, qf[j], partner), kr = __shfl_sync(~0u, kf[j], partner);
      qo[j] = round_bf16(qf[j] * c[j] + sign * qr * sn[j]);
      ko[j] = round_bf16(kf[j] * c[j] + sign * kr * sn[j]);
      vf[j] = round_bf16(vf[j]);
      pq[j] += qo[j];
      pk[j] += ko[j];
      pv[j] += vf[j];
    }
    const size_t dst = (plane + i) * D + l * 4;
    st4(qb + dst, qo);
    st4(kb + dst, ko);
    st4(vb + dst, vf);
    if constexpr (CC) {
      const size_t cdst = ((size_t)(row0 + i) * heads + h) * D + l * 4;
      st4(qc + cdst, qo);
      st4(kc + cdst, ko);
      st4(vc + cdst, vf);
    }
  };
  for (int i = w; i < n; i += 2 * NWARP) {
    const int i2 = i + NWARP;
    const bool two = i2 < n;
    float qf[4], kf[4], vf[4], qf2[4], kf2[4], vf2[4];
    const size_t src = (row0 + i) * ld + h * D + l * 4, src2 = (row0 + (two ? i2 : i)) * ld + h * D + l * 4;
    ld4(q + src, qf);
    ld4(k + src, kf);
    ld4(v + src, vf);
    ld4(q + src2, qf2);
    ld4(k + src2, kf2);
    ld4(v + src2, vf2);
    const float s = s1[row0 + i], s2 = s1[row0 + (two ? i2 : i)];
    float c[4] = {1.f, 1.f, 1.f, 1.f}, sn[4] = {0.f, 0.f, 0.f, 0.f}, c2[4] = {1.f, 1.f, 1.f, 1.f},
          sn2[4] = {0.f, 0.f, 0.f, 0.f};
    if (rot) {
      ld4(cosb + (int64_t)(row0 + i) * ROT + l * 4, c);
      ld4(sinb + (int64_t)(row0 + i) * ROT + l * 4, sn);
      ld4(cosb + (int64_t)(row0 + (two ? i2 : i)) * ROT + l * 4, c2);
      ld4(sinb + (int64_t)(row0 + (two ? i2 : i)) * ROT + l * 4, sn2);
    }
    row(i, qf, kf, vf, s, c, sn);
    if (two) row(i2, qf2, kf2, vf2, s2, c2, sn2);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sp[0][w][l * 4 + j] = pq[j];
    sp[1][w][l * 4 + j] = pk[j];
    sp[2][w][l * 4 + j] = pv[j];
  }
  __syncthreads();
  const float inv = 1.f / (float)n;
  for (int e = threadIdx.x; e < 3 * D; e += NWARP * 32) {
    const int a = e / D, d = e % D;
    float acc = 0.f;
#pragma unroll
    for (int ww = 0; ww < NWARP; ++ww) acc += sp[a][ww][d];
    float* dst = a == 0 ? qp : (a == 1 ? kp : vp);
    dst[((size_t)h * n_tiles + ti) * D + d] = acc * inv;
  }
}

// 8-bit attention inputs: E4M3 V with one scale per (head, tile) plus the running per-head maximum of those scales; Q and
// K either BF16 (QK16) or E4M3 with static per-layer scales (|x| <= sqrt(128) * max|w| after the RMSNorm).
// WB > 0 also writes compact [rows, H, D] copies for the dense passes: Q and K as BF16 (WB 1) or E4M3 with the static
// scales (WB 2), V as BF16.
template <int NWARP, int TILE, bool QK16, int WB>
__global__ void __launch_bounds__(NWARP * 32) qkv_prologue8(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, int64_t ld,
    const float* __restrict__ s1, const int* __restrict__ tile_row0, const int* __restrict__ vbs,
    const bf16* __restrict__ wq, const bf16* __restrict__ wk, const bf16* __restrict__ cosb,
    const bf16* __restrict__ sinb, float inv_sq, float inv_sk, void* __restrict__ qb, void* __restrict__ kb,
    uint8_t* __restrict__ vb, float* __restrict__ vs_blk, float* __restrict__ vs_head, float* __restrict__ qp,
    float* __restrict__ kp, float* __restrict__ vp, int heads, int64_t rows_pad, int n_tiles, int n_ext, float eps,
    void* __restrict__ qc, void* __restrict__ kc, bf16* __restrict__ vc) {
  constexpr int D = 128, ROT = 96, HALF_LANES = 12;
  __shared__ float sp[3][NWARP][D];
  __shared__ bf16 sv[TILE][D];  // V is already rounded to BF16, so staging it as BF16 is exact
  __shared__ float sm[32];
  const int h = blockIdx.x % heads, ti = blockIdx.x / heads;
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  const int row0 = tile_row0[ti], n = vbs[ti];
  const int partner = l < HALF_LANES ? l + HALF_LANES : (l < 2 * HALF_LANES ? l - HALF_LANES : l);
  const float sign = l < HALF_LANES ? -1.f : 1.f;
  const bool rot = l < 2 * HALF_LANES;
  float wqv[4], wkv[4];
  ld4(wq + l * 4, wqv);
  ld4(wk + l * 4, wkv);
  float pq[4] = {0.f, 0.f, 0.f, 0.f}, pk[4] = {0.f, 0.f, 0.f, 0.f}, pv[4] = {0.f, 0.f, 0.f, 0.f};
  float vmax0 = 0.f, vmax1 = 0.f;  // V scale per 64-key block: a 128-token tile has two
  const size_t plane = (size_t)h * rows_pad + (size_t)ti * TILE;
  for (int i = w; i < n; i += NWARP) {
    const int64_t r = row0 + i;
    const size_t src = r * ld + h * D + l * 4;
    float qf[4], kf[4], vf[4];
    ld4(q + src, qf);
    ld4(k + src, kf);
    ld4(v + src, vf);
    const float s = s1[r];
    float sumq = 0.f, sumk = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qf[j] *= s;
      kf[j] *= s;
      vf[j] *= s;
      sumq += qf[j] * qf[j];
      sumk += kf[j] * kf[j];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      sumq += __shfl_xor_sync(~0u, sumq, o);
      sumk += __shfl_xor_sync(~0u, sumk, o);
    }
    const float rq = rsqrtf(sumq / D + eps), rk = rsqrtf(sumk / D + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qf[j] *= rq * wqv[j];
      kf[j] *= rk * wkv[j];
    }
    float c[4] = {1.f, 1.f, 1.f, 1.f}, sn[4] = {0.f, 0.f, 0.f, 0.f};
    if (rot) {
      ld4(cosb + r * ROT + l * 4, c);
      ld4(sinb + r * ROT + l * 4, sn);
    }
    float qo[4], ko[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float qr = __shfl_sync(~0u, qf[j], partner), kr = __shfl_sync(~0u, kf[j], partner);
      qo[j] = round_bf16(qf[j] * c[j] + sign * qr * sn[j]);
      ko[j] = round_bf16(kf[j] * c[j] + sign * kr * sn[j]);
      vf[j] = round_bf16(vf[j]);
      pq[j] += qo[j];
      pk[j] += ko[j];
      pv[j] += vf[j];
      if (i < 64) vmax0 = fmaxf(vmax0, fabsf(vf[j]));
      else vmax1 = fmaxf(vmax1, fabsf(vf[j]));
      sv[i][l * 4 + j] = __float2bfloat16_rn(vf[j]);
    }
    const size_t dst = (plane + i) * D + l * 4;
    if constexpr (QK16) {
      st4(reinterpret_cast<bf16*>(qb) + dst, qo);
      st4(reinterpret_cast<bf16*>(kb) + dst, ko);
    } else {
      *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(qb) + dst) =
          fp8x4(qo[0] * inv_sq, qo[1] * inv_sq, qo[2] * inv_sq, qo[3] * inv_sq);
      *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(kb) + dst) =
          fp8x4(ko[0] * inv_sk, ko[1] * inv_sk, ko[2] * inv_sk, ko[3] * inv_sk);
    }
    if constexpr (WB > 0) {
      const size_t c = (size_t)r * heads * D + h * D + l * 4;
      if constexpr (WB == 1) {
        st4(reinterpret_cast<bf16*>(qc) + c, qo);
        st4(reinterpret_cast<bf16*>(kc) + c, ko);
      } else {
        *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(qc) + c) =
            fp8x4(qo[0] * inv_sq, qo[1] * inv_sq, qo[2] * inv_sq, qo[3] * inv_sq);
        *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(kc) + c) =
            fp8x4(ko[0] * inv_sk, ko[1] * inv_sk, ko[2] * inv_sk, ko[3] * inv_sk);
      }
      st4(vc + c, vf);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sp[0][w][l * 4 + j] = pq[j];
    sp[1][w][l * 4 + j] = pk[j];
    sp[2][w][l * 4 + j] = pv[j];
  }
  vmax0 = block_reduce<NWARP, true>(vmax0, sm);
  vmax1 = TILE > 64 ? block_reduce<NWARP, true>(vmax1, sm) : 0.f;
  const float svb0 = fmaxf(vmax0 / kFp8Max, 1e-30f), svb1 = fmaxf(vmax1 / kFp8Max, 1e-30f);
  if (threadIdx.x == 0) {
    // n_ext counts 64-key blocks of the padded sequence
    vs_blk[(size_t)h * n_ext + ti * (TILE / 64)] = svb0;
    atomicMax(reinterpret_cast<int*>(vs_head + h), __float_as_int(svb0));
    if (TILE > 64) {
      vs_blk[(size_t)h * n_ext + ti * 2 + 1] = svb1;
      atomicMax(reinterpret_cast<int*>(vs_head + h), __float_as_int(svb1));
    }
  }
  for (int i = w; i < n; i += NWARP) {
    const size_t dst = (plane + i) * D + l * 4;
    const float inv_v = 1.f / (i < 64 ? svb0 : svb1);
    *reinterpret_cast<uint32_t*>(vb + dst) =
        fp8x4(__bfloat162float(sv[i][l * 4]) * inv_v, __bfloat162float(sv[i][l * 4 + 1]) * inv_v,
              __bfloat162float(sv[i][l * 4 + 2]) * inv_v, __bfloat162float(sv[i][l * 4 + 3]) * inv_v);
  }
  const float inv = 1.f / (float)n;
  for (int e = threadIdx.x; e < 3 * D; e += NWARP * 32) {
    const int a2 = e / D, d = e % D;
    float acc = 0.f;
#pragma unroll
    for (int ww = 0; ww < NWARP; ++ww) acc += sp[a2][ww][d];
    float* dst = a2 == 0 ? qp : (a2 == 1 ? kp : vp);
    dst[((size_t)h * n_tiles + ti) * D + d] = acc * inv;
  }
}

struct Spans {
  int n;
  int start[4];
  int end[4];
  int k[4];
};

__device__ __forceinline__ uint32_t sortable(float f) {
  const uint32_t u = __float_as_uint(f);
  return u ^ ((u >> 31) ? 0xffffffffu : 0x80000000u);
}

// One warp per (head, query tile): prefix tiles, then each span's top-k, ascending, straight into the kernel's index list.
// With `split` the dense passes cover the prefix queries and the prefix keys: prefix query tiles select nothing and
// the other lists hold video tiles only.
template <int PER>
__global__ void __launch_bounds__(128) topk_rows(const float* __restrict__ scores, int n, int n_ext, int prefix,
                                                 Spans sp, int rows, int* __restrict__ idx, int* __restrict__ num,
                                                 bool split) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (row >= rows) return;
  const int hh = row / n, q = row % n;
  int* out = idx + ((size_t)hh * n_ext + q) * n;
  const float* srow = scores + (size_t)row * n;
  int* cnt_out = num + (size_t)hh * n_ext + q;
  if (q < prefix) {  // prefix queries are dense
    if (!split)
      for (int c = l; c < n; c += 32) out[c] = c;
    if (l == 0) *cnt_out = split ? 0 : n;
    return;
  }
  int cnt = 0;
  if (!split) {
    for (int c = l; c < prefix; c += 32) out[c] = c;
    cnt = prefix;
  }
  const uint32_t lt = (1u << l) - 1u;
  for (int s = 0; s < sp.n; ++s) {
    const int start = sp.start[s], len = sp.end[s] - start, k = sp.k[s];
    if (k >= len) {
      for (int c = l; c < len; c += 32) out[cnt + c] = start + c;
      cnt += len;
      continue;
    }
    uint32_t key[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int col = p * 32 + l;
      key[p] = col < len ? sortable(srow[start + col]) : 0u;
    }
    uint32_t th = 0;
    for (int b = 31; b >= 0; --b) {
      const uint32_t cand = th | (1u << b);
      int c = 0;
#pragma unroll
      for (int p = 0; p < PER; ++p) c += key[p] >= cand;
      if (__reduce_add_sync(~0u, c) >= k) th = cand;
    }
    int c = 0;
#pragma unroll
    for (int p = 0; p < PER; ++p) c += key[p] > th;
    const int need = k - __reduce_add_sync(~0u, c);
    int seen = 0;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int col = p * 32 + l;
      const bool valid = col < len, gt = valid && key[p] > th, eq = valid && key[p] == th;
      const uint32_t em = __ballot_sync(~0u, eq);
      const bool keep = gt || (eq && seen + __popc(em & lt) < need);
      seen += __popc(em);
      const uint32_t km = __ballot_sync(~0u, keep);
      if (keep) out[cnt + __popc(km & lt)] = start + col;
      cnt += __popc(km);
    }
  }
  if (l == 0) *cnt_out = cnt;
}

// One warp per (head, pair of adjacent query tiles): both tiles' selections, merged into one ascending union list with a
// membership byte per entry (bit 0: first tile, bit 1: second tile), for the pair attention kernel.
template <int PER>
__global__ void __launch_bounds__(128) topk_pairs(const float* __restrict__ scores, int n, int n_ext, int prefix,
                                                  Spans sp, int pairs_total, int* __restrict__ u_idx,
                                                  uint8_t* __restrict__ u_mem, int* __restrict__ u_num) {
  const int gp = blockIdx.x * 4 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  const int n_pairs = n_ext / 2;
  if (gp >= pairs_total) return;
  const int hh = gp / n_pairs, pr = gp % n_pairs;
  const int r0 = 2 * pr, r1 = 2 * pr + 1;
  const bool v0 = r0 < n, v1 = r1 < n;                       // pad tiles select nothing
  const bool d0 = v0 && r0 < prefix, d1 = v1 && r1 < prefix;  // prefix query tiles are dense
  const float* s0 = scores + ((size_t)hh * n + (v0 ? r0 : 0)) * n;
  const float* s1 = scores + ((size_t)hh * n + (v1 ? r1 : 0)) * n;
  int* oi = u_idx + (size_t)gp * n;
  uint8_t* om = u_mem + (size_t)gp * n;
  const uint32_t lt = (1u << l) - 1u;
  int cnt = 0;
  const uint8_t pm = (uint8_t)((v0 ? 1 : 0) | (v1 ? 2 : 0));
  if (pm) {
    for (int c = l; c < prefix; c += 32) {
      oi[c] = c;
      om[c] = pm;
    }
    cnt = prefix;
  }
  for (int s = 0; s < sp.n; ++s) {
    const int start = sp.start[s], len = sp.end[s] - start, k = sp.k[s];
    uint32_t key0[PER], key1[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int col = p * 32 + l;
      key0[p] = (v0 && col < len) ? sortable(s0[start + col]) : 0u;
      key1[p] = (v1 && col < len) ? sortable(s1[start + col]) : 0u;
    }
    const bool all0 = d0 || (v0 && k >= len), all1 = d1 || (v1 && k >= len);
    uint32_t th0 = 0, th1 = 0;
    if (v0 && !all0) {
      for (int b = 31; b >= 0; --b) {
        const uint32_t cand = th0 | (1u << b);
        int c = 0;
#pragma unroll
        for (int p = 0; p < PER; ++p) c += key0[p] >= cand;
        if (__reduce_add_sync(~0u, c) >= k) th0 = cand;
      }
    }
    if (v1 && !all1) {
      for (int b = 31; b >= 0; --b) {
        const uint32_t cand = th1 | (1u << b);
        int c = 0;
#pragma unroll
        for (int p = 0; p < PER; ++p) c += key1[p] >= cand;
        if (__reduce_add_sync(~0u, c) >= k) th1 = cand;
      }
    }
    int g0 = 0, g1 = 0;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      g0 += key0[p] > th0;
      g1 += key1[p] > th1;
    }
    const int need0 = k - __reduce_add_sync(~0u, g0), need1 = k - __reduce_add_sync(~0u, g1);
    int seen0 = 0, seen1 = 0;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int col = p * 32 + l;
      const bool valid = col < len;
      bool keep0 = false, keep1 = false;
      if (all0) {
        keep0 = valid;
      } else if (v0) {
        const bool eq = valid && key0[p] == th0;
        const uint32_t em = __ballot_sync(~0u, eq);
        keep0 = (valid && key0[p] > th0) || (eq && seen0 + __popc(em & lt) < need0);
        seen0 += __popc(em);
      }
      if (all1) {
        keep1 = valid;
      } else if (v1) {
        const bool eq = valid && key1[p] == th1;
        const uint32_t em = __ballot_sync(~0u, eq);
        keep1 = (valid && key1[p] > th1) || (eq && seen1 + __popc(em & lt) < need1);
        seen1 += __popc(em);
      }
      const bool any = keep0 || keep1;
      const uint32_t am = __ballot_sync(~0u, any);
      if (any) {
        const int pos = cnt + __popc(am & lt);
        oi[pos] = start + col;
        om[pos] = (uint8_t)((keep0 ? 1 : 0) | (keep1 ? 2 : 0));
      }
      cnt += __popc(am);
    }
  }
  if (l == 0) u_num[gp] = cnt;
}

// out[row] = attention (heads-first, padded) + gate * compressed branch, back to compact rows, FP8 with a row scale.
// With MERGE the sparse output (ob, log2-domain LSE lse2 [H, rows_pad]) is combined with a dense output on the compact
// rows (o1 [rows, H, D]; natural-log LSE lse1a [H, n1] for rows < n1, lse1b [H, rows - n1] for the rest):
// o = (e^lse1 o1 + e^lse2 o2) / (e^lse1 + e^lse2).
template <int W, bool GATE, bool FP4, bool MERGE>
__global__ void __launch_bounds__(W / 8) attn_gate_quant(const bf16* __restrict__ ob, const bf16* __restrict__ g,
                                                         const float* __restrict__ sg, const bf16* __restrict__ oc,
                                                         const int* __restrict__ slot, int64_t rows_pad, int n_tiles,
                                                         int tile, fp8s* __restrict__ a, uint8_t* __restrict__ asf,
                                                         float* __restrict__ sa, const bf16* __restrict__ o1,
                                                         const float* __restrict__ lse1a, const float* __restrict__ lse1b,
                                                         int n1, const float* __restrict__ lse2) {
  constexpr int NW = W / 8 / 32, D = 128, H = W / D;
  __shared__ float sm[32];
  const int64_t r = blockIdx.x;
  const int t = threadIdx.x, h = t / (D / 8), d0 = (t % (D / 8)) * 8;
  const int sl = slot[r];
  float o[8];
  ld8(ob + ((size_t)h * rows_pad + sl) * D + d0, o);
  if constexpr (MERGE) {
    float od[8];
    ld8(o1 + ((size_t)r * H + h) * D + d0, od);
    const float l1 = (r < n1 ? lse1a[(size_t)h * n1 + r] : lse1b[(size_t)h * (gridDim.x - n1) + (r - n1)]) *
                     1.4426950408889634f;
    // prefix rows select no sparse keys (split lists), so whatever the sparse kernel left there is ignored
    const float l2 = r < n1 ? -INFINITY : lse2[(size_t)h * rows_pad + sl];
    const float m = fmaxf(l1, l2);
    const float w1 = m == -INFINITY ? 0.f : exp2f(l1 - m), w2 = m == -INFINITY ? 0.f : exp2f(l2 - m);
    const float inv = (w1 + w2) > 0.f ? 1.f / (w1 + w2) : 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = (w1 * od[i] + (w2 > 0.f ? w2 * o[i] : 0.f)) * inv;
  }
  if constexpr (GATE) {
    float gv[8], cv[8];
    ld8(g + r * W + t * 8, gv);
    ld8(oc + ((size_t)h * n_tiles + sl / tile) * D + d0, cv);
    const float s = sg[r];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] += cv[i] * (s * gv[i]);
  }
  float am = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(o[i]));
  am = block_reduce<NW, true>(am, sm);
  if constexpr (FP4) {
    const float s = fmaxf(am / kFp4Range, kMinScale4);
    st_fp4x8(reinterpret_cast<uint8_t*>(a) + r * (W / 2), asf, o, 1.f / s, r, t, W / 16);
    if (t == 0) sa[r] = s;
  } else {
    const float s = fmaxf(am / kFp8Max, kMinScale);
    st_fp8x8(a + r * W + t * 8, o, 1.f / s);
    if (t == 0) sa[r] = s;
  }
}

// Compact BF16 V [rows, H, D] to E4M3 with one scale per head (vs_head, from the prologue).
__global__ void __launch_bounds__(256) v_to_fp8(const bf16* __restrict__ v, const float* __restrict__ vs_head, int heads,
                                                int64_t n8, uint8_t* __restrict__ o) {
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n8) return;
  const int h = (int)((i / 16) % heads);  // 16 groups of 8 per 128-wide head
  float f[8];
  ld8(v + i * 8, f);
  const float inv = 1.f / vs_head[h];
  uint2 w;
  w.x = fp8x4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
  w.y = fp8x4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
  *reinterpret_cast<uint2*>(o + i * 8) = w;
}

// value * silu(gate) of the packed fc_in output (value half first), FP8 with a row scale.
template <int F, bool FP4>
__global__ void __launch_bounds__(F / 16) swiglu_quant(const bf16* __restrict__ u, const float* __restrict__ su,
                                                       fp8s* __restrict__ o, uint8_t* __restrict__ osf,
                                                       float* __restrict__ so) {
  constexpr int NW = F / 16 / 32;
  __shared__ float sm[32];
  const int64_t r = blockIdx.x;
  const int t = threadIdx.x;
  const bf16* row = u + r * (2 * F);
  const float s = su[r];
  float y[2][8];
  float am = 0.f;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    float a[8], b[8];
    ld8(row + t * 16 + c * 8, a);
    ld8(row + F + t * 16 + c * 8, b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float gg = s * b[i];
      y[c][i] = (s * a[i]) * (gg / (1.f + __expf(-gg)));
      am = fmaxf(am, fabsf(y[c][i]));
    }
  }
  am = block_reduce<NW, true>(am, sm);
  if constexpr (FP4) {
    const float sc = fmaxf(am / kFp4Range, kMinScale4);
    st_fp4x16(reinterpret_cast<uint8_t*>(o) + r * (F / 2), osf, y, 1.f / sc, r, t, F / 16);
    if (t == 0) so[r] = sc;
  } else {
    const float sc = fmaxf(am / kFp8Max, kMinScale), inv = 1.f / sc;
    st_fp8x8(o + r * F + t * 16, y[0], inv);
    st_fp8x8(o + r * F + t * 16 + 8, y[1], inv);
    if (t == 0) so[r] = sc;
  }
}

template <typename T>
T* ptr(const torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
T* opt(const std::optional<torch::Tensor>& t) {
  return t.has_value() ? reinterpret_cast<T*>(t->data_ptr()) : nullptr;
}

}  // namespace

// Branch and/or norm over [rows, 5376]; passing no y skips the residual, passing no nw skips the norm.
void resid_norm_quant_op(torch::Tensor x, std::optional<torch::Tensor> y, std::optional<torch::Tensor> sy,
                         std::optional<torch::Tensor> gate, torch::Tensor idx, std::optional<torch::Tensor> nw,
                         std::optional<torch::Tensor> shift, std::optional<torch::Tensor> scale, torch::Tensor xo,
                         std::optional<torch::Tensor> h, std::optional<torch::Tensor> hsf,
                         std::optional<torch::Tensor> hs, double eps) {
  constexpr int C = 5376;
  TORCH_CHECK(x.size(-1) == C && x.is_contiguous());
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = x.numel() / C;
  const int64_t gate_ld = gate.has_value() ? gate->stride(0) : 0, tab_ld = scale.has_value() ? scale->stride(0) : 0;
  const bool branch = y.has_value(), norm = nw.has_value();
  const bool fp4 = hsf.has_value();
  constexpr int RPC = 1;  // 2 rows per CTA measured 0.48 vs 0.29 ms (register pressure halves occupancy)
#define H3_LAUNCH(B, N, Q)                                                                                                 \
  resid_norm_quant<C, B, N, Q, RPC><<<(rows + RPC - 1) / RPC, C / 8, 0, stream>>>(                                         \
      ptr<bf16>(x), opt<bf16>(y), opt<float>(sy), opt<bf16>(gate), ptr<int>(idx), opt<bf16>(nw), opt<bf16>(shift),         \
      opt<bf16>(scale), gate_ld, tab_ld, ptr<bf16>(xo), opt<fp8s>(h), opt<uint8_t>(hsf), opt<float>(hs), (float)eps,       \
      (int64_t)rows)
  if (branch && norm) {
    if (fp4) H3_LAUNCH(true, true, true);
    else H3_LAUNCH(true, true, false);
  } else if (branch) H3_LAUNCH(true, false, false);
  else if (fp4) H3_LAUNCH(false, true, true);
  else H3_LAUNCH(false, true, false);
#undef H3_LAUNCH
}

void qkv_prologue_op(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor s1, torch::Tensor tile_row0,
                     torch::Tensor vbs, torch::Tensor wq, torch::Tensor wk, torch::Tensor cosb, torch::Tensor sinb,
                     torch::Tensor qb, torch::Tensor kb, torch::Tensor vb, torch::Tensor qp, torch::Tensor kp,
                     torch::Tensor vp, int64_t heads, int64_t n_tiles, int64_t tile, double eps,
                     std::optional<torch::Tensor> qc, std::optional<torch::Tensor> kc, std::optional<torch::Tensor> vc) {
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(cosb.size(-1) == 96 && sinb.size(-1) == 96);
  TORCH_CHECK(qc.has_value() == kc.has_value() && kc.has_value() == vc.has_value());
  auto launch = [&](auto cc) {
    qkv_prologue<8, decltype(cc)::value><<<n_tiles * heads, 256, 0, stream>>>(
        ptr<bf16>(q), ptr<bf16>(k), ptr<bf16>(v), q.stride(0), ptr<float>(s1), ptr<int>(tile_row0), ptr<int>(vbs),
        ptr<bf16>(wq), ptr<bf16>(wk), ptr<bf16>(cosb), ptr<bf16>(sinb), ptr<bf16>(qb), ptr<bf16>(kb), ptr<bf16>(vb),
        ptr<float>(qp), ptr<float>(kp), ptr<float>(vp), (int)heads, qb.size(-2), (int)n_tiles, (int)tile, (float)eps,
        qc ? ptr<bf16>(*qc) : nullptr, kc ? ptr<bf16>(*kc) : nullptr, vc ? ptr<bf16>(*vc) : nullptr);
  };
  if (qc) launch(std::true_type{});
  else launch(std::false_type{});
}

void qkv_prologue8_op(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor s1, torch::Tensor tile_row0,
                      torch::Tensor vbs, torch::Tensor wq, torch::Tensor wk, torch::Tensor cosb, torch::Tensor sinb,
                      double inv_sq, double inv_sk, torch::Tensor qb, torch::Tensor kb, torch::Tensor vb,
                      torch::Tensor vs_blk, torch::Tensor vs_head, torch::Tensor qp, torch::Tensor kp, torch::Tensor vp,
                      int64_t heads, int64_t n_tiles, int64_t tile, double eps, std::optional<torch::Tensor> qc,
                      std::optional<torch::Tensor> kc, std::optional<torch::Tensor> vc) {
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(tile == 64 || tile == 128, "qkv_prologue8: 64- or 128-token tiles");
  const bool qk16 = qb.scalar_type() == at::kBFloat16;
#define H3_P8(T, Q16, WB)                                                                                                  \
  qkv_prologue8<8, T, Q16, WB><<<n_tiles * heads, 256, 0, stream>>>(                                                       \
      ptr<bf16>(q), ptr<bf16>(k), ptr<bf16>(v), q.stride(0), ptr<float>(s1), ptr<int>(tile_row0), ptr<int>(vbs),          \
      ptr<bf16>(wq), ptr<bf16>(wk), ptr<bf16>(cosb), ptr<bf16>(sinb), (float)inv_sq, (float)inv_sk, qb.data_ptr(),        \
      kb.data_ptr(), ptr<uint8_t>(vb), ptr<float>(vs_blk), ptr<float>(vs_head), ptr<float>(qp), ptr<float>(kp),          \
      ptr<float>(vp), (int)heads, vb.size(-2), (int)n_tiles, (int)vs_blk.size(-1), (float)eps,                         \
      qc.has_value() ? qc->data_ptr() : nullptr, kc.has_value() ? kc->data_ptr() : nullptr, opt<bf16>(vc))
  const int wb = !qc.has_value() ? 0 : (qc->scalar_type() == at::kBFloat16 ? 1 : 2);
  TORCH_CHECK(wb == 0 || (tile == 128 && kc.has_value() && vc.has_value()), "qkv_prologue8: write-back needs tile 128");
  if (tile == 64) {
    if (qk16) H3_P8(64, true, 0);
    else H3_P8(64, false, 0);
  } else if (wb == 1) {
    if (qk16) H3_P8(128, true, 1);
    else H3_P8(128, false, 1);
  } else if (wb == 2) {
    if (qk16) H3_P8(128, true, 2);
    else H3_P8(128, false, 2);
  } else if (qk16) {
    H3_P8(128, true, 0);
  } else {
    H3_P8(128, false, 0);
  }
#undef H3_P8
}

torch::Tensor vsa_fp8_op(torch::Tensor q8, torch::Tensor k8, torch::Tensor v8, torch::Tensor q2k_idx,
                         torch::Tensor q2k_num, torch::Tensor vbs, double qk_coef, torch::Tensor v_scale_blk,
                         torch::Tensor v_scale_head);

void topk_rows_op(torch::Tensor scores, int64_t n, int64_t n_ext, int64_t prefix, std::vector<int64_t> starts,
                  std::vector<int64_t> ends, std::vector<int64_t> ks, torch::Tensor idx, torch::Tensor num,
                  bool split) {
  const at::cuda::OptionalCUDAGuard guard(device_of(scores));
  auto stream = at::cuda::getCurrentCUDAStream();
  Spans sp{};
  TORCH_CHECK(starts.size() <= 4 && starts.size() == ends.size() && ks.size() == ends.size());
  sp.n = (int)starts.size();
  int longest = 0;
  for (int i = 0; i < sp.n; ++i) {
    sp.start[i] = (int)starts[i];
    sp.end[i] = (int)ends[i];
    sp.k[i] = (int)ks[i];
    longest = std::max(longest, (int)(ends[i] - starts[i]));
  }
  const int rows = scores.numel() / n;
  const int grid = (rows + 3) / 4;
#define H3_TOPK(P)                                                                                                         \
  topk_rows<P><<<grid, 128, 0, stream>>>(ptr<float>(scores), (int)n, (int)n_ext, (int)prefix, sp, rows, ptr<int>(idx),     \
                                         ptr<int>(num), split)
  if (longest <= 24 * 32) H3_TOPK(24);
  else if (longest <= 48 * 32) H3_TOPK(48);
  else if (longest <= 96 * 32) H3_TOPK(96);
  else {
    TORCH_CHECK(longest <= 160 * 32, "topk_rows: span too long");
    H3_TOPK(160);
  }
#undef H3_TOPK
}

void topk_pairs_op(torch::Tensor scores, int64_t n, int64_t n_ext, int64_t prefix, std::vector<int64_t> starts,
                   std::vector<int64_t> ends, std::vector<int64_t> ks, torch::Tensor u_idx, torch::Tensor u_mem,
                   torch::Tensor u_num) {
  const at::cuda::OptionalCUDAGuard guard(device_of(scores));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(n_ext % 2 == 0);
  Spans sp{};
  sp.n = (int)starts.size();
  int longest = 0;
  for (int i = 0; i < sp.n; ++i) {
    sp.start[i] = (int)starts[i];
    sp.end[i] = (int)ends[i];
    sp.k[i] = (int)ks[i];
    longest = std::max(longest, (int)(ends[i] - starts[i]));
  }
  const int heads = (int)(scores.numel() / (n * n));
  const int pairs = heads * (int)(n_ext / 2);
  const int grid = (pairs + 3) / 4;
#define H3_TOPKP(P)                                                                                                        \
  topk_pairs<P><<<grid, 128, 0, stream>>>(ptr<float>(scores), (int)n, (int)n_ext, (int)prefix, sp, pairs, ptr<int>(u_idx),  \
                                          ptr<uint8_t>(u_mem), ptr<int>(u_num))
  if (longest <= 24 * 32) H3_TOPKP(24);
  else if (longest <= 48 * 32) H3_TOPKP(48);
  else {
    TORCH_CHECK(longest <= 96 * 32, "topk_pairs: span too long");
    H3_TOPKP(96);
  }
#undef H3_TOPKP
}

torch::Tensor vsa_pair_op(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor u_idx, torch::Tensor u_mem,
                          torch::Tensor u_num, torch::Tensor vbs, double sm_scale, std::optional<torch::Tensor> v_scale_blk,
                          std::optional<torch::Tensor> v_scale_head, std::optional<torch::Tensor> lse);

void attn_gate_quant_op(torch::Tensor ob, std::optional<torch::Tensor> g, std::optional<torch::Tensor> sg,
                        std::optional<torch::Tensor> oc, torch::Tensor slot, int64_t n_tiles, int64_t tile,
                        torch::Tensor a, std::optional<torch::Tensor> asf, torch::Tensor sa, int64_t rows,
                        std::optional<torch::Tensor> o1, std::optional<torch::Tensor> lse1a,
                        std::optional<torch::Tensor> lse1b, std::optional<torch::Tensor> lse2) {
  constexpr int W = 7168;
  const at::cuda::OptionalCUDAGuard guard(device_of(ob));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool fp4 = asf.has_value(), merge = o1.has_value();
  TORCH_CHECK(!merge || (lse1a.has_value() && lse1b.has_value() && lse2.has_value() && o1->is_contiguous()));
  const int n1 = merge ? (int)lse1a->size(-1) : 0;
#define H3_LAUNCH(G, Q, M)                                                                                                 \
  attn_gate_quant<W, G, Q, M><<<(int)rows, W / 8, 0, stream>>>(                                                            \
      ptr<bf16>(ob), opt<bf16>(g), opt<float>(sg), opt<bf16>(oc), ptr<int>(slot), ob.size(-2), (int)n_tiles, (int)tile,    \
      ptr<fp8s>(a), opt<uint8_t>(asf), ptr<float>(sa), opt<bf16>(o1), opt<float>(lse1a), opt<float>(lse1b), n1,         \
      opt<float>(lse2))
#define H3_LAUNCH_M(G, Q)                                                                                                  \
  if (merge) H3_LAUNCH(G, Q, true);                                                                                        \
  else H3_LAUNCH(G, Q, false)
  if (g.has_value()) {
    if (fp4) { H3_LAUNCH_M(true, true); }
    else { H3_LAUNCH_M(true, false); }
  } else if (fp4) { H3_LAUNCH_M(false, true); }
  else { H3_LAUNCH_M(false, false); }
#undef H3_LAUNCH_M
#undef H3_LAUNCH
}

void swiglu_quant_op(torch::Tensor u, torch::Tensor su, torch::Tensor o, std::optional<torch::Tensor> osf, torch::Tensor so) {
  constexpr int F = 14336;
  const at::cuda::OptionalCUDAGuard guard(device_of(u));
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(u.size(-1) == 2 * F);
  const int rows = (int)(u.numel() / (2 * F));
  if (osf.has_value())
    swiglu_quant<F, true><<<rows, F / 16, 0, stream>>>(ptr<bf16>(u), ptr<float>(su), ptr<fp8s>(o), opt<uint8_t>(osf), ptr<float>(so));
  else
    swiglu_quant<F, false><<<rows, F / 16, 0, stream>>>(ptr<bf16>(u), ptr<float>(su), ptr<fp8s>(o), nullptr, ptr<float>(so));
}

void v_to_fp8_op(torch::Tensor v, torch::Tensor vs_head, torch::Tensor o) {
  const at::cuda::OptionalCUDAGuard guard(device_of(v));
  const int64_t n8 = v.numel() / 8;
  v_to_fp8<<<(unsigned)((n8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      ptr<bf16>(v), ptr<float>(vs_head), (int)vs_head.numel(), n8, ptr<uint8_t>(o));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("resid_norm_quant", &resid_norm_quant_op);
  m.def("qkv_prologue", &qkv_prologue_op);
  m.def("topk_rows", &topk_rows_op);
  m.def("attn_gate_quant", &attn_gate_quant_op);
  m.def("swiglu_quant", &swiglu_quant_op);
  m.def("qkv_prologue8", &qkv_prologue8_op);
  m.def("vsa_fp8", &vsa_fp8_op);
  m.def("topk_pairs", &topk_pairs_op);
  m.def("vsa_pair", &vsa_pair_op);
  m.def("v_to_fp8", &v_to_fp8_op);
}

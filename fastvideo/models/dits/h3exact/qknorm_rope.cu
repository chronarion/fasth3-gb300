// Bit-exact fusion of MiniMax-H3's per-head q/k RMSNorm (torch nn.RMSNorm over D = 128 on CUDA = ATen's
// vectorized_layer_norm_kernel<bf16, float, rms_norm=true>) with FastVideo's exact RoPE prefix (exact.py
// _rope_prefix_kernel: rotate the first R channels, pass the rest).
//
// ATen, per row of 128 BF16 values (vec_size 4, one warp of lanes 0..31 reads one vector each; other warps add 0):
//   s_lane = fma(x3, x3, fma(x2, x2, fma(x1, x1, fma(x0, x0, 0))))      (sigma2 + val * val, contracted by nvcc)
//   warp tree: s += shfl_down(s, 16), 8, 4, 2, 1                         (lane 0 holds the row sum)
//   ms = s / 128 (div.rn; exact for a power of two); rstd = rsqrt.approx(ms + eps)
//   y = bf16(gamma * (rstd * x))                                         (two rounded float multiplies)
// exact RoPE on y (BF16) for d < R:  a = bf16(y[d] * cos[d]); b = bf16(p[d] * sin[d]); out = bf16(a + b)
//   with p[d] = -y[d + R/2] (d < R/2) or y[d - R/2]; cos / sin rounded to BF16 first; d >= R: out = y.
// One warp per (token, head) row; q and k in one launch (blockIdx.y), written in place. Used by the full 50-step model
// (FASTVIDEO_H3_FAST=0) with FASTVIDEO_MINIMAX_H3_EXACT_KERNELS including rope; see h3exact/__init__.py.
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

template <int R>
__global__ void __launch_bounds__(256) qknorm_rope_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ wq,
    const __nv_bfloat16* __restrict__ wk, const __nv_bfloat16* __restrict__ cos, const __nv_bfloat16* __restrict__ sin,
    int64_t rows, int heads, int64_t q_ts, int64_t q_hs, int64_t k_ts, int64_t k_hs, int64_t c_ts, float eps) {
  const int lane = threadIdx.x & 31;
  const int64_t row = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) return;
  const bool isk = blockIdx.y == 1;
  __nv_bfloat16* x = isk ? k : q;
  const __nv_bfloat16* w = isk ? wk : wq;
  const int64_t t = row / heads, h = row % heads;
  __nv_bfloat16* xr = x + t * (isk ? k_ts : q_ts) + h * (isk ? k_hs : q_hs);
  // lane holds channels 4*lane .. 4*lane+3
  const uint2 raw = *reinterpret_cast<const uint2*>(xr + 4 * lane);
  const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float xv[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) xv[i] = __bfloat162float(xb[i]);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) s = __fmaf_rn(xv[i], xv[i], s);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
  s = __shfl_sync(0xffffffffu, s, 0);
  const float rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(s, 128.f), eps));
  const uint2 wraw = *reinterpret_cast<const uint2*>(w + 4 * lane);
  const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wraw);
  float y[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) y[i] = bf(__fmul_rn(__bfloat162float(wb[i]), __fmul_rn(rstd, xv[i])));
  // RoPE partner: channel d +- R/2 lives (R/2)/4 lanes away
  constexpr int HALF = R / 2, LS = HALF / 4;
  float p[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float up = __shfl_down_sync(0xffffffffu, y[i], LS);   // y[d + HALF]
    const float dn = __shfl_up_sync(0xffffffffu, y[i], LS);     // y[d - HALF]
    p[i] = (4 * lane < HALF) ? -up : dn;
  }
  __align__(8) __nv_bfloat16 ob[4];
  if (4 * lane < R) {
    const __nv_bfloat16* cr = cos + t * c_ts + 4 * lane;
    const __nv_bfloat16* sr = sin + t * c_ts + 4 * lane;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float a = bf(__fmul_rn(y[i], __bfloat162float(cr[i])));
      const float b = bf(__fmul_rn(p[i], __bfloat162float(sr[i])));
      ob[i] = __float2bfloat16_rn(__fadd_rn(a, b));
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) ob[i] = __float2bfloat16_rn(y[i]);
  }
  *reinterpret_cast<uint2*>(xr + 4 * lane) = *reinterpret_cast<const uint2*>(ob);
}

// q, k: [B, S, H, 128] BF16 with unit channel stride (any token / head strides); cos, sin: [S, R] BF16 contiguous rows
void qknorm_rope(torch::Tensor q, torch::Tensor k, torch::Tensor wq, torch::Tensor wk, torch::Tensor cos,
                 torch::Tensor sin, double eps) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.size(0) == 1 && k.size(0) == 1, "qknorm_rope: [1, S, H, D]");
  TORCH_CHECK(q.size(3) == 128 && q.stride(3) == 1 && k.stride(3) == 1 && q.sizes() == k.sizes(), "qknorm_rope: D");
  TORCH_CHECK(cos.size(-1) == 96 && cos.stride(-1) == 1 && sin.strides() == cos.strides(), "qknorm_rope: R = 96");
  const int64_t S = q.size(1), H = q.size(2), rows = S * H;
  const dim3 grid((unsigned) ((rows + 7) / 8), 2);
  qknorm_rope_kernel<96><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (__nv_bfloat16*) q.data_ptr(), (__nv_bfloat16*) k.data_ptr(), (const __nv_bfloat16*) wq.data_ptr(),
      (const __nv_bfloat16*) wk.data_ptr(), (const __nv_bfloat16*) cos.data_ptr(), (const __nv_bfloat16*) sin.data_ptr(),
      rows, (int) H, q.stride(1), q.stride(2), k.stride(1), k.stride(2), cos.stride(0), (float) eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("qknorm_rope", &qknorm_rope); }

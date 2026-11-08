// SPDX-License-Identifier: Apache-2.0
// Derived from FastVideo's fastvideo-kernel/csrc/attention/block_sparse_kernel_blackwell.cuh
// (https://github.com/hao-ai-lab/FastVideo, Apache-2.0); the changes are described below.
// 8-bit variants of fastvideo-kernel's VSA block-sparse forward (block_sparse_kernel_blackwell.cuh, blk64, BHSD).
//
// Same warp specialization, pipeline and per-q-tile top-k walk; PV always runs as tcgen05.mma.ws.kind::f8f6f4 on E4M3 P
// and V. QK^T either stays BF16 (QK16: Q and K as the BF16 kernel loads them) or runs on E4M3 Q and K (sm_103a has no
// kind::i8; on captured H3 layers the E4M3 QK^T is what dominates the error, 2-5% per head, against ~1% each for P
// and V):
//   * E4M3 Q and K carry one static scale per layer each (|x| <= sqrt(128) * max|w| after the RMSNorm); qk_coef folds
//     them and the softmax scale into the exponent.
//   * V carries one scale per (head, 64-key block). P for block b is formed as exp2(... + log2(sv_b / sv_head) + P_SHIFT)
//     so the MMA sees P * sv_b / sv_head in E4M3 range; the row sum divides that factor back out; the epilogue multiplies
//     by sv_head.
//   * The lazy-rescale threshold drops from 8 to RESCALE_LOG2 so P (<= 2^(P_SHIFT + RESCALE_LOG2)) stays below 448.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include "vsa_primitives.cuh"

namespace vsa_fp8 {

constexpr int BLOCK = 64;
constexpr int M_TILE = 64;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 256;
constexpr int BLOCKS_PER_KTILE = 4;
constexpr int HEAD_DIM = 128;
constexpr int ROW_BYTES = 128;                         // one E4M3 head row = one 128-byte swizzle atom row
constexpr int Q_TILE_BYTES = 2 * M_TILE * ROW_BYTES;   // room for a BF16 Q tile (two 64-column subtiles)
constexpr int KV_RING_SLOT_BYTES = K_TILE * ROW_BYTES;  // 32 KB: a whole K or V tile per slot
constexpr int BLK_BYTES = BLOCK * ROW_BYTES;           // 8 KB
constexpr int NUM_KV_STAGES = 4;
constexpr int S_COLS = 128;
constexpr int K_ATOMS = ROW_BYTES / 32;                // QK^T MMAs per K-tile (K = 32 per kind::f8f6f4 MMA)
constexpr int PV_ATOMS = 128 / 32;                     // PV MMAs per K-tile: 128 keys per lane half
constexpr int P_COLS = 128 / 4;                        // E4M3 P: 4 per 32-bit TMEM column
constexpr int SPLIT_P_ATOM = 2;                        // PV atoms 0-1 start once the first half of P is in TMEM
constexpr int SPLIT_P_COL = SPLIT_P_ATOM * 8;
constexpr int EX2_FRG_PAIRS = 16;
constexpr int EX2_FRG_CNT = S_COLS / 32;
constexpr int EX2_FREQ = 16;
constexpr int EX2_RES = 0;  // GB300 doubles the SFU ex2 rate: hardware ex2 for every pair is 1.5-5% faster than emulating 4 of 16
constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;
constexpr int STATS = 2 * M_TILE;
constexpr int STAT_REGIONS = 3;
constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;
constexpr int SUB_COLS_BF16 = 64;                      // output staging stays BF16 (same epilogue as the BF16 kernel)
constexpr float P_SHIFT = 6.f;
constexpr int RESCALE_LOG2 = 2;

extern __shared__ __align__(1024) uint8_t fmha8_smem[];

__device__ __forceinline__ void mma_ws_f8_ss(uint32_t lead, uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
                                             uint32_t idesc, bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "@q tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%1], %2, %3, %5, p, 0;\n\t"
      "}\n" ::"r"(lead),
      "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(enable_input_d ? 1u : 0u), "r"(idesc));
}

__device__ __forceinline__ void mma_ws_f16_ss(uint32_t lead, uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
                                              uint32_t idesc, bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "@q tcgen05.mma.ws.cta_group::1.kind::f16 [%1], %2, %3, %5, p, 0;\n\t"
      "}\n" ::"r"(lead),
      "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(enable_input_d ? 1u : 0u), "r"(idesc));
}

__device__ __forceinline__ void mma_ws_f8_ts(uint32_t lead, uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b,
                                             uint32_t idesc, bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "@q tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%1], [%2], %3, %5, p, 0;\n\t"
      "}\n" ::"r"(lead),
      "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(enable_input_d ? 1u : 0u), "r"(idesc));
}

__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
  uint32_t r;
  asm("{\n\t.reg .b16 lo, hi;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n\t"
      "mov.b32 %0, {lo, hi};\n\t}"
      : "=r"(r)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}

__device__ __forceinline__ void tcgen05_st_32x32b_x16_u(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
               "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(taddr),
               "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]),
               "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]));
}

struct WorkItem {
  int sample, head, mtile0, mtile1, global_mtile0, global_mtile1, num_kv_blocks;
  int num_kv_blocks_mt[2];
};

__device__ __forceinline__ WorkItem decode_workitem(int workitem_id, int num_heads, int num_blocks,
                                                    int packed_mtiles_per_seq, unsigned long long magic0,
                                                    unsigned long long magic2, const int* q2k_num) {
  WorkItem it;
  const int per_sample = num_heads * packed_mtiles_per_seq;
  it.sample = (int)fdiv((unsigned)workitem_id, magic0);
  const int rem = workitem_id - it.sample * per_sample;
  it.head = (int)fdiv((unsigned)rem, magic2);
  const int p = rem - it.head * packed_mtiles_per_seq;
  it.mtile0 = 2 * p;
  it.mtile1 = 2 * p + 1;
  it.global_mtile0 = (it.sample * num_heads + it.head) * num_blocks + it.mtile0;
  it.global_mtile1 = it.global_mtile0 + 1;
  it.num_kv_blocks_mt[0] = q2k_num[it.global_mtile0];
  it.num_kv_blocks_mt[1] = q2k_num[it.global_mtile1];
  it.num_kv_blocks = max(max(it.num_kv_blocks_mt[0], it.num_kv_blocks_mt[1]), 1);
  return it;
}

template <typename F>
__device__ __forceinline__ bool next_workitem(int& workitem_id, uint64_t* clc_full, uint64_t* clc_empty,
                                              uint32_t* clc_response, int& clc_stage, uint32_t& clc_phase) {
  ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                 clc_stage, clc_phase, elect_one_sync());
  clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
  if (!next.valid) return false;
  workitem_id = next.n_tile;
  return true;
}

template <bool QK16>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_fp8_vsa_kernel(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k,
                    const __grid_constant__ CUtensorMap tmap_v, const __grid_constant__ CUtensorMap tmap_o,
                    int seqlen, int num_heads, int num_samples, int num_blocks, int packed_mtiles_per_seq, int max_kv,
                    unsigned long long magic0, unsigned long long magic2, const int* __restrict__ q2k_idx,
                    const int* __restrict__ q2k_num, const int* __restrict__ variable_block_sizes,
                    float qk_coef, const float* __restrict__ v_scale_blk, const float* __restrict__ v_scale_head,
                    float* __restrict__ lse_out) {
#if !defined(__CUDA_ARCH__) || ((__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)) || \
                                (__CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)))
  uint8_t* sQ0 = fmha8_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = {sQ0, sQ1};
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * KV_RING_SLOT_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = {sO0, sO1};
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar = full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q = full_bar_q + 2;
  uint64_t* full_bar_spo = empty_bar_q + 2;
  uint64_t* empty_bar_spo = full_bar_spo + 2;
  uint64_t* full_bar_o_acc = empty_bar_spo + 2;
  uint64_t* full_bar_alpha = full_bar_o_acc + 2;
  uint64_t* full_bar_l = full_bar_alpha + 2;
  uint64_t* full_bar_p_last = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* full_bar_o_epi = empty_bar_alpha_and_l + 2;
  uint64_t* empty_bar_o_epi = full_bar_o_epi + 2;
  uint64_t* clc_full = empty_bar_o_epi + 2;
  uint64_t* clc_empty = clc_full + CLC_STAGES;
  uint32_t* clc_response =
      reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  const uint32_t tmem_base = *tmem_slot;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < NUM_KV_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), 1);
    }
    for (int i = 0; i < 2; ++i) {
      mbarrier_init(smem_ptr_u32(&full_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_spo[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_spo[i]), 256);
      mbarrier_init(smem_ptr_u32(&full_bar_o_acc[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_alpha[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_alpha_and_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_p_last[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_o_epi[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_o_epi[i]), 1);
    }
#pragma unroll
    for (int s = 0; s < CLC_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
      mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS);
    }
#pragma unroll
    for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<48>();
    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic2,
                                          q2k_num);
      const int bh = it.sample * num_heads + it.head;
      const int global_mtile[2] = {it.global_mtile0, it.global_mtile1};
      const int mtile[2] = {it.mtile0, it.mtile1};
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;

      int window_start[2] = {-(1 << 30), -(1 << 30)};
      int kv_block_id_cache[2] = {0, 0};
      auto get_kv_block_id = [&](int mtile_idx, int j) -> int {
        if (j >= window_start[mtile_idx] + 32) {
          window_start[mtile_idx] = j & ~31;
          const int cnt = it.num_kv_blocks_mt[mtile_idx];
          const int idx = max(0, min(window_start[mtile_idx] + lane, cnt - 1));
          kv_block_id_cache[mtile_idx] = (cnt > 0) ? q2k_idx[global_mtile[mtile_idx] * max_kv + idx] : 0;
        }
        return __shfl_sync(0xffffffffu, kv_block_id_cache[mtile_idx], j & 31);
      };
      // K tile: blocks 0..3 stacked as 256 K-major rows. V tile: lane half 0 reads keys of blocks 0, 1 against N
      // columns 0-127 and lane half 1 reads blocks 2, 3 against N columns 128-255, so the MN-major layout puts block
      // 2 one LBO (8 KB) after block 0: offsets 0, 16, 8, 24 KB for blocks 0, 1, 2, 3.
      auto load_kv = [&](int mtile_idx, int ktile_idx, const CUtensorMap* map, bool v_layout) {
        const int kv_stage = kv_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        int kv_blk[BLOCKS_PER_KTILE];
#pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
          kv_blk[blk] = get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + blk);
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
#pragma unroll
          for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk) {
            const int pos = v_layout ? ((blk & 1) * 2 + (blk >> 1)) : blk;
            tma_load_3d(smem_ptr_u32(sKV + kv_stage * KV_RING_SLOT_BYTES + pos * BLK_BYTES), map,
                        smem_ptr_u32(&full_bar[kv_stage]), 0, kv_blk[blk] * BLOCK, bh);
          }
        }
      };
      // BF16 K: one slot per 64-column subtile, blocks stacked as 256 K-major rows of 128 bytes (as the BF16 kernel).
      auto load_k16 = [&](int mtile_idx, int ktile_idx, int sub) {
        const int kv_stage = kv_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        int kv_blk[BLOCKS_PER_KTILE];
#pragma unroll
        for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
          kv_blk[blk] = get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + blk);
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
#pragma unroll
          for (int blk = 0; blk < BLOCKS_PER_KTILE; ++blk)
            tma_load_4d(smem_ptr_u32(sKV + kv_stage * KV_RING_SLOT_BYTES + blk * BLK_BYTES), &tmap_k,
                        smem_ptr_u32(&full_bar[kv_stage]), 0, kv_blk[blk] * BLOCK, sub, bh);
        }
      };
      auto load_k = [&](int m, int kt) {
        if constexpr (QK16) {
          load_k16(m, kt, 0);
          load_k16(m, kt, 1);
        } else {
          load_kv(m, kt, &tmap_k, false);
        }
      };
      auto load_v = [&](int m, int kt) { load_kv(m, kt, &tmap_v, true); };

      load_k(0, 0);
      load_k(1, 0);
#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        if (elect_one_sync()) {
          if constexpr (QK16) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), 2 * M_TILE * ROW_BYTES);
            tma_load_4d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]), 0, mtile[m] * BLOCK, 0, bh);
            tma_load_4d(smem_ptr_u32(sQ[m] + M_TILE * ROW_BYTES), &tmap_q, smem_ptr_u32(&full_bar_q[m]), 0,
                        mtile[m] * BLOCK, 1, bh);
          } else {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), M_TILE * ROW_BYTES);
            tma_load_3d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]), 0, mtile[m] * BLOCK, bh);
          }
        }
      }
      q_empty_ph.advance();
      for (int kt = 0; kt + 1 < num_k_tiles; ++kt) {
        load_v(0, kt);
        load_k(0, kt + 1);
        load_v(1, kt);
        load_k(1, kt + 1);
      }
      load_v(0, num_k_tiles - 1);
      load_v(1, num_k_tiles - 1);
      if (!next_workitem<int>(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  } else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();
    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    // QK^T: dtype F32 (1), A/B BF16 (1) or E4M3 (0). PV: F32, E4M3 (0), V read MN-major.
    const uint32_t idesc_qk = QK16 ? make_idesc_table44(M_TILE, K_TILE, 1, 1, 1, false, false)
                                   : make_idesc_table44(M_TILE, K_TILE, 1, 0, 0, false, false);
    const uint32_t idesc_pv = make_idesc_table44(M_TILE, 2 * HEAD_DIM, 1, 0, 0, false, true);
    const uint64_t desc_q0 = build_smem_desc_blackwell(smem_ptr_u32(sQ0), 1024, 16, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), 1024, 16, SmemSwizzleBlackwell::B128);
    const uint64_t desc_v0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), 1024, 2 * BLK_BYTES / 2 / 1,
                                                       SmemSwizzleBlackwell::B128);
    constexpr uint64_t KV_DESC_DELTA = KV_RING_SLOT_BYTES >> 4;
    constexpr uint64_t Q_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic2,
                                          q2k_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;

      auto bmm1 = [&](int i) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint64_t da = desc_q0 + (uint64_t)i * Q_DESC_DELTA;
        if constexpr (QK16) {
#pragma unroll
          for (int sub = 0; sub < 2; ++sub) {
            const int slot = kv_ph.get_stage();
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
            kv_ph.advance();
            const uint64_t db = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA;
            const uint64_t dq = da + (uint64_t)sub * ((M_TILE * ROW_BYTES) >> 4);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)  // K = 16 BF16 per MMA, four per 128-byte subtile
              mma_ws_f16_ss(lead, s_tmem_addr, dq + 2 * ki, db + 2 * ki, idesc_qk, (sub | ki) != 0);
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
          }
        } else {
          const int slot = kv_ph.get_stage();
          mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
          kv_ph.advance();
          const uint64_t db = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA;
#pragma unroll
          for (int ki = 0; ki < K_ATOMS; ++ki) mma_ws_f8_ss(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, ki != 0);
          tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
      };

      auto bmm2 = [&](int i, bool first_ktile, auto last_c) {
        constexpr bool last_ktile = decltype(last_c)::value;
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        const int slot = kv_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
        kv_ph.advance();
        const uint64_t dv = desc_v0 + (uint64_t)slot * KV_DESC_DELTA;
#pragma unroll
        for (int a = 0; a < PV_ATOMS; ++a) {
          if (a == SPLIT_P_ATOM) mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
          // atom a: keys 32a..32a+31 of each lane half: block a/2 (and a/2 + 2), rows 32*(a%2) of that block.
          const uint64_t off = (uint64_t)((a >> 1) * (2 * BLK_BYTES) + (a & 1) * (32 * ROW_BYTES)) >> 4;
          mma_ws_f8_ts(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dv + off, idesc_pv, (!first_ktile) || (a != 0));
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
        if constexpr (last_ktile) tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_o_acc[i]));
      };

#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
        bmm1(i);
      }
      for (int k = 0; k + 1 < num_k_tiles; ++k) {
#pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          bmm2(i, (k == 0), std::false_type{});
          bmm1(i);
        }
        spo_ph.advance();
      }
#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
      q_ph.advance();
#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
        bmm2(i, (num_k_tiles == 1), std::true_type{});
      }
      spo_ph.advance();
      if (!next_workitem<int>(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  } else if (warp_id == W_EPI) {
    setmaxnreg_dec<48>();
    PhaseTracker<1> full_o_ph;
    if (elect_one_sync()) {
#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic2,
                                          q2k_num);
      const int mtile[2] = {it.mtile0, it.mtile1};
#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        if (elect_one_sync()) {
          tma_store_4d(&tmap_o, 0, it.sample * num_heads + it.head, mtile[m] * BLOCK, 0,
                       smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m])));
          cp_async_bulk_commit_group();
        }
      }
      if (elect_one_sync()) {
        cp_async_bulk_wait_group_read<1>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        cp_async_bulk_wait_group_read<0>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
      }
      full_o_ph.advance();
      if (!next_workitem<int>(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  } else if (warp_id == W_SCHED) {
    setmaxnreg_dec<48>();
    int prod_stage = 0;
    uint32_t prod_phase = 1;
    int cons_stage = 0;
    uint32_t cons_phase = 0;
    while (true) {
      if (lane == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
      __syncwarp();
      clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), 16);
      if (lane == 0) clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]), smem_ptr_u32(&clc_full[prod_stage]));
      advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                     cons_stage, cons_phase,
                                                                                     elect_one_sync());
      clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
      if (!next.valid) break;
    }
    for (int s = 0; s < CLC_STAGES; ++s) {
      if (lane == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
      __syncwarp();
      advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
    }
  } else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<80>();
    const int corr_warp_id = warp_id - W_CORR0;
    const int corr_tid = corr_warp_id * 32 + lane;
    const int corr_row = corr_tid & 63;
    const bool kv_half0 = corr_tid < 64;
    PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;
#pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic2,
                                          q2k_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;
      const float sl2 = qk_coef;
      const float vs_head = v_scale_head[it.sample * num_heads + it.head];
#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
      }
      alpha_ph.advance();
      for (int k = 1; k < num_k_tiles; ++k) {
#pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_alpha[i]), alpha_ph.get_phase());
          const float alpha = alpha_and_l_smem[(i * STAT_REGIONS + 0) * STATS + corr_tid];
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
          const bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
          if (!skip) {
            const uint32_t o_tmem_addr =
                tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
            const float2 alpha2 = f32x2_splat(alpha);
#pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
#pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], alpha2);
              tcgen05_st_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            }
            tcgen05_wait_st();
            tcgen05_fence_before_thread_sync();
          }
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        }
        alpha_ph.advance();
      }
#pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
        const float l_own = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
        const float m_own = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + corr_tid];
        const float l_par = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + (corr_tid ^ 64)];
        const float m_par = alpha_and_l_smem[(i * STAT_REGIONS + 2) * STATS + (corr_tid ^ 64)];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        const float d = (m_own - m_par) * sl2;
        const float beta_lo = ex2_approx_f32(-fabsf(d));
        const float beta_own = (d >= 0.f) ? 1.f : beta_lo;
        const float beta_par = (d >= 0.f) ? beta_lo : 1.f;
        const float l_tot = beta_own * l_own + beta_par * l_par;
        const float scale_own = (l_tot > 0.f) ? beta_own * rcp_approx_ftz_f32(l_tot) * vs_head : 0.f;
        if (lse_out != nullptr && (corr_tid >> 6) == 0) {
          // log2-domain LSE of the row (both halves combined; l carries 2^P_SHIFT, the V-block factors divided out)
          const float m_tot = (d >= 0.f) ? m_own : m_par;
          const int q_row = (i == 0 ? it.mtile0 : it.mtile1) * BLOCK + (corr_tid & 63);
          lse_out[(size_t)(it.sample * num_heads + it.head) * seqlen + q_row] =
              l_tot > 0.f ? m_tot * sl2 + __log2f(l_tot) - P_SHIFT : -INFINITY;
        }
        const float2 scale2 = f32x2_splat(scale_own);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
        if (!kv_half0) {
#pragma unroll
          for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
            uint32_t o_regs[16];
            tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            float2* o2 = reinterpret_cast<float2*>(o_regs);
#pragma unroll
            for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], scale2);
            const int s = c0 / SUB_COLS_BF16;
            const int v_base = (c0 % SUB_COLS_BF16) / 8;
            __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
#pragma unroll
            for (int vv = 0; vv < 2; ++vv) {
              const int v = v_base + vv;
              uint4 packed;
              packed.x = cvt_f32x2_to_bf16x2(o2[vv * 4 + 0].x, o2[vv * 4 + 0].y);
              packed.y = cvt_f32x2_to_bf16x2(o2[vv * 4 + 1].x, o2[vv * 4 + 1].y);
              packed.z = cvt_f32x2_to_bf16x2(o2[vv * 4 + 2].x, o2[vv * 4 + 2].y);
              packed.w = cvt_f32x2_to_bf16x2(o2[vv * 4 + 3].x, o2[vv * 4 + 3].y);
              *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
            }
          }
        }
        bar_sync<9>(128);
        if (kv_half0) {
#pragma unroll
          for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
            uint32_t o_regs[16];
            tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            float2* o2 = reinterpret_cast<float2*>(o_regs);
#pragma unroll
            for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], scale2);
            const int s = c0 / SUB_COLS_BF16;
            const int v_base = (c0 % SUB_COLS_BF16) / 8;
            __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
#pragma unroll
            for (int vv = 0; vv < 2; ++vv) {
              const int v = v_base + vv;
              __nv_bfloat16* dst = &so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8];
              uint4 h1 = *reinterpret_cast<uint4*>(dst);
              o2[vv * 4 + 0] = fadd2(o2[vv * 4 + 0], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.x)));
              o2[vv * 4 + 1] = fadd2(o2[vv * 4 + 1], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.y)));
              o2[vv * 4 + 2] = fadd2(o2[vv * 4 + 2], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.z)));
              o2[vv * 4 + 3] = fadd2(o2[vv * 4 + 3], __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&h1.w)));
              uint4 packed;
              packed.x = cvt_f32x2_to_bf16x2(o2[vv * 4 + 0].x, o2[vv * 4 + 0].y);
              packed.y = cvt_f32x2_to_bf16x2(o2[vv * 4 + 1].x, o2[vv * 4 + 1].y);
              packed.z = cvt_f32x2_to_bf16x2(o2[vv * 4 + 2].x, o2[vv * 4 + 2].y);
              packed.w = cvt_f32x2_to_bf16x2(o2[vv * 4 + 3].x, o2[vv * 4 + 3].y);
              *reinterpret_cast<uint4*>(dst) = packed;
            }
          }
        }
        bar_sync<9>(128);
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        fence_proxy_async_shared();
        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();
      if (!next_workitem<int>(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  } else {
    setmaxnreg_inc<192>();
    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    const int half = warp_in_group >> 1;
    const int sm_tid = warp_in_group * 32 + lane;
    const uint32_t alpha_slot_u32 = smem_ptr_u32(&alpha_and_l_smem[(m_tile * STAT_REGIONS + 0) * STATS + sm_tid]);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_blocks, packed_mtiles_per_seq, magic0, magic2,
                                          q2k_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;
      const int bh = it.sample * num_heads + it.head;
      const float sl2 = qk_coef;
      const float lv_head = __log2f(v_scale_head[bh]);
      const float* vs_row = v_scale_blk + (size_t)bh * num_blocks;
      float m_run = -INFINITY, l_run = 0.f;
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();

      // Per 32-K-tile window: each lane caches the vbs thresholds and log2(sv_b / sv_head) of its K-tile's two blocks.
      int thr_window = -(1 << 30);
      int thr_cache0 = BLOCK, thr_cache1 = BLOCK;
      float lr_cache0 = 0.f, lr_cache1 = 0.f;
      const int gqb_mt = (m_tile == 0) ? it.global_mtile0 : it.global_mtile1;
      const int cnt = it.num_kv_blocks_mt[m_tile];
      auto get_block_info = [&](int k, int& t0, int& t1, float& lr0, float& lr1) {
        if (k >= thr_window + 32) {
          thr_window = k & ~31;
          const int b0 = (thr_window + lane) * BLOCKS_PER_KTILE + 2 * half;
          const int id0 = (b0 < cnt) ? q2k_idx[gqb_mt * max_kv + b0] : -1;
          const int id1 = (b0 + 1 < cnt) ? q2k_idx[gqb_mt * max_kv + b0 + 1] : -1;
          thr_cache0 = id0 >= 0 ? variable_block_sizes[id0] : 0;
          thr_cache1 = id1 >= 0 ? variable_block_sizes[id1] : 0;
          lr_cache0 = id0 >= 0 ? __log2f(fmaxf(vs_row[id0], 1e-30f)) - lv_head : 0.f;
          lr_cache1 = id1 >= 0 ? __log2f(fmaxf(vs_row[id1], 1e-30f)) - lv_head : 0.f;
        }
        t0 = __shfl_sync(0xffffffffu, thr_cache0, k & 31);
        t1 = __shfl_sync(0xffffffffu, thr_cache1, k & 31);
        lr0 = __shfl_sync(0xffffffffu, lr_cache0, k & 31);
        lr1 = __shfl_sync(0xffffffffu, lr_cache1, k & 31);
      };
      auto softmax_step = [&](auto is_first_c, int k) {
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        int vbs_thr0, vbs_thr1;
        float lr0, lr1;
        get_block_info(k, vbs_thr0, vbs_thr1, lr0, lr1);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
        uint32_t s_regs[S_COLS];
        float hw[S_COLS / 32];  // ld.red: the max of each 32-column chunk
#pragma unroll
        for (int c0 = 0; c0 < S_COLS; c0 += 32)
          tcgen05_ld_red_max_32x32b_x32(s_tmem_addr + (uint32_t)c0, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]),
                                        hw[c0 / 32]);
        float* scores = reinterpret_cast<float*>(s_regs);
        float2* scores2 = reinterpret_cast<float2*>(s_regs);
        tcgen05_fence_before_thread_sync();
        // A block with masked keys reduces over its masked scores in software; the others use the ld.red maxima.
        float new_m = m_run;
        if (vbs_thr0 < 64) {
          mask_s_row_r2p<false, 64>(scores, 0, 0, vbs_thr0);
          float ra = -INFINITY, rb = -INFINITY;
#pragma unroll
          for (int j = 0; j < 64; j += 4) {
            ra = fmaxf(fmaxf(ra, scores[j + 0]), scores[j + 1]);
            rb = fmaxf(fmaxf(rb, scores[j + 2]), scores[j + 3]);
          }
          new_m = fmaxf(new_m, fmaxf(ra, rb));
        } else {
          new_m = fmaxf(new_m, fmaxf(hw[0], hw[1]));
        }
        if (vbs_thr1 < 64) {
          mask_s_row_r2p<false, 64>(scores + 64, 0, 0, vbs_thr1);
          float ra = -INFINITY, rb = -INFINITY;
#pragma unroll
          for (int j = 64; j < 128; j += 4) {
            ra = fmaxf(fmaxf(ra, scores[j + 0]), scores[j + 1]);
            rb = fmaxf(fmaxf(rb, scores[j + 2]), scores[j + 3]);
          }
          new_m = fmaxf(new_m, fmaxf(ra, rb));
        } else {
          new_m = fmaxf(new_m, fmaxf(hw[2], hw[3]));
        }

        new_m = fmaxf(new_m, -FLT_MAX);
        float alpha = 0.0f;
        if constexpr (!IS_FIRST) {
          const float acc_scale = (m_run - new_m) * sl2;
          if (acc_scale >= -(float)RESCALE_LOG2) {
            new_m = m_run;
            alpha = 1.0f;
          } else {
            alpha = ex2_approx_f32(acc_scale);
          }
          sts_f32(alpha_slot_u32, alpha);
        }
        mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        const float2 scale2 = f32x2_splat(sl2);
        const float base = -new_m * sl2 + P_SHIFT;
        const float2 nm0 = f32x2_splat(base + lr0), nm1 = f32x2_splat(base + lr1);
#pragma unroll
        for (int c = 0; c < S_COLS / 2; ++c) {
          const float2 a2 = ffma2(scores2[c], scale2, c < S_COLS / 4 ? nm0 : nm1);
          const int jj = c / EX2_FRG_PAIRS;
          const int kk = 2 * (c % EX2_FRG_PAIRS);
          const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1);
          scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y)) : ex2_emu_f32x2(a2.x, a2.y);
        }
        uint32_t p_regs[P_COLS];
#pragma unroll
        for (int c = 0; c < P_COLS; ++c)
          p_regs[c] = e4m3x4(scores2[2 * c].x, scores2[2 * c].y, scores2[2 * c + 1].x, scores2[2 * c + 1].y);
        tcgen05_st_32x32b_x16_u(s_tmem_addr, &p_regs[0]);
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        tcgen05_st_32x32b_x16_u(s_tmem_addr + SPLIT_P_COL, &p_regs[SPLIT_P_COL]);
        tcgen05_wait_st();
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        spo_ph.advance();

        float2 la = make_float2(0.f, 0.f), lb = make_float2(0.f, 0.f), lc = make_float2(0.f, 0.f), ld = make_float2(0.f, 0.f);
#pragma unroll
        for (int c = 0; c < S_COLS / 4; c += 2) {
          la = fadd2(la, scores2[c]);
          lb = fadd2(lb, scores2[c + 1]);
          lc = fadd2(lc, scores2[S_COLS / 4 + c]);
          ld = fadd2(ld, scores2[S_COLS / 4 + c + 1]);
        }
        const float l0 = (la.x + la.y) + (lb.x + lb.y), l1 = (lc.x + lc.y) + (ld.x + ld.y);
        l_run = (IS_FIRST ? 0.f : l_run * alpha) + l0 * ex2_approx_f32(-lr0) + l1 * ex2_approx_f32(-lr1);
        m_run = new_m;
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
      };
      if (num_k_tiles > 0) softmax_step(std::true_type{}, 0);
      for (int k = 1; k < num_k_tiles; ++k) softmax_step(std::false_type{}, k);
      alpha_and_l_smem[(m_tile * STAT_REGIONS + 1) * STATS + sm_tid] = l_run;
      alpha_and_l_smem[(m_tile * STAT_REGIONS + 2) * STATS + sm_tid] = m_run;
      mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
      if (!next_workitem<int>(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  }
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
#endif
}

struct Args {
  bool qk16;         // Q and K are BF16 [B*H, S, 128]; else E4M3
  const void* q;
  const void* k;
  const uint8_t* v;
  __nv_bfloat16* o;  // BF16 [B, H, S, 128]
  const int* q2k_idx;
  const int* q2k_num;
  const int* variable_block_sizes;
  float qk_coef;               // softmax scale * log2(e) * (E4M3 Q scale * K scale)
  const float* v_scale_blk;    // [B*H, num_blocks]
  const float* v_scale_head;   // [B*H]
  float* lse;                  // optional [B*H, S] log2-domain LSE per query row
  int batch, num_heads, seqlen, num_blocks, max_kv;
};

inline cudaError_t make_fp8_map(CUtensorMap* map, const uint8_t* base, long planes, int seqlen) {
  uint64_t gd[3] = {(uint64_t)ROW_BYTES, (uint64_t)seqlen, (uint64_t)planes};
  uint64_t gs[2] = {(uint64_t)ROW_BYTES, (uint64_t)seqlen * ROW_BYTES};
  uint32_t bd[3] = {(uint32_t)ROW_BYTES, (uint32_t)BLOCK, 1u};
  uint32_t es[3] = {1u, 1u, 1u};
  return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<uint8_t*>(base), gd, gs, bd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

inline cudaError_t make_bf16_map(CUtensorMap* map, const void* base, long planes, int seqlen) {
  // [64 cols][S][2 column halves][planes], box 64 cols x 64 rows: one 128-byte swizzled subtile per load.
  uint64_t gd[4] = {64u, (uint64_t)seqlen, 2u, (uint64_t)planes};
  uint64_t gs[3] = {(uint64_t)HEAD_DIM * 2u, 128u, (uint64_t)seqlen * HEAD_DIM * 2u};
  uint32_t bd[4] = {64u, (uint32_t)BLOCK, 1u, 1u};
  uint32_t es[4] = {1u, 1u, 1u, 1u};
  return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), gd, gs, bd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

inline cudaError_t launch(const Args& a, cudaStream_t stream) {
  if (a.num_blocks % 2 || a.seqlen != a.num_blocks * BLOCK || a.max_kv < 1) return cudaErrorInvalidValue;
  const int B = a.batch, H = a.num_heads, S = a.seqlen;
  const int packed_mtiles_per_seq = a.num_blocks / 2;
  const int total_work = B * H * packed_mtiles_per_seq;
  CUtensorMap tq, tk, tv, to;
  if (a.qk16) {
    if (make_bf16_map(&tq, a.q, (long)B * H, S) != cudaSuccess || make_bf16_map(&tk, a.k, (long)B * H, S) != cudaSuccess)
      return cudaErrorInvalidValue;
  } else if (make_fp8_map(&tq, (const uint8_t*)a.q, (long)B * H, S) != cudaSuccess ||
             make_fp8_map(&tk, (const uint8_t*)a.k, (long)B * H, S) != cudaSuccess) {
    return cudaErrorInvalidValue;
  }
  if (make_fp8_map(&tv, a.v, (long)B * H, S) != cudaSuccess) return cudaErrorInvalidValue;
  {
    // BF16 output map, as the BF16 kernel's BHSD Q/O map: [64 cols][B*H][S][2 subtiles].
    uint64_t gd[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)((long)B * H), (uint64_t)S, 2u};
    uint64_t gs[3] = {(uint64_t)((long)S * HEAD_DIM) * 2u, (uint64_t)HEAD_DIM * 2u, (uint64_t)SUB_COLS_BF16 * 2u};
    uint32_t bd[4] = {(uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, 2u};
    uint32_t es[4] = {1u, 1u, 1u, 1u};
    if (cuTensorMapEncodeTiled(&to, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, a.o, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  const size_t smem = (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_RING_SLOT_BYTES +
                      (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16) + (2 * NUM_KV_STAGES + 22) * 8 +
                      (size_t)CLC_STAGES * (2 * 8 + 16) + 16 + 8 + (size_t)2 * STAT_REGIONS * STATS * sizeof(float) + 256;
  auto kfn = a.qk16 ? fmha_fp8_vsa_kernel<true> : fmha_fp8_vsa_kernel<false>;
  cudaError_t e = cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e != cudaSuccess) return e;
  const unsigned long long magic0 = make_magic((unsigned)(H * packed_mtiles_per_seq));
  const unsigned long long magic2 = make_magic((unsigned)packed_mtiles_per_seq);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(total_work, 1, 1);
  cfg.blockDim = dim3(N_WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kfn, tq, tk, tv, to, S, H, B, a.num_blocks, packed_mtiles_per_seq, a.max_kv, magic0,
                            magic2, a.q2k_idx, a.q2k_num, a.variable_block_sizes, a.qk_coef, a.v_scale_blk,
                            a.v_scale_head, a.lse);
}

}  // namespace vsa_fp8

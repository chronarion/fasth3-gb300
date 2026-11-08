// SPDX-License-Identifier: Apache-2.0
// Derived from FastVideo's fastvideo-kernel/csrc/attention/block_sparse_kernel_blackwell.cuh
// (https://github.com/hao-ai-lab/FastVideo, Apache-2.0); the changes are described below.
// Pair variant of fastvideo-kernel's VSA block-sparse forward for 64-token tiles (BHSD, BF16).
//
// The 64-token kernel gives each query tile its own K/V stream on 64-row .ws MMAs, so it moves twice FA4's K/V bytes per
// FLOP and is L2-bound (tensor pipe ~54% busy). Here two adjacent 64-token query tiles form one 128-row tile on full
// M=128 MMAs (the structure of the 128-token kernel) and walk the UNION of their top-k lists in K-tiles of two 64-key
// blocks. A per-block membership byte (bit 0: first tile selected it, bit 1: second tile) masks the rows of the tile
// that did not select the block, so every row attends exactly its own selection: same math as the 64-token kernel.
// On H3 the union is ~1.3x a single list, so this trades 1.3x MMA work for 0.65x K/V traffic (measured: slower for
// t2v's 64-token tiles). It also runs 128-token VSA exactly (each 128-key block as two 64-key blocks selected by both
// halves), and with PV8 it runs PV as kind::f8f6f4 on E4M3 P and V: V carries one scale per (head, 64-key block),
// folded into P's exponent and divided back out of the row sum; the epilogue multiplies by the per-head maximum.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include "vsa_primitives.cuh"

namespace vsa_pair {

constexpr int KB = 64;                      // key block (the selection granularity)
constexpr int M_TILE = 128;                 // one pair of 64-token query tiles
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;                 // two key blocks per K-tile
constexpr int BLOCKS_PER_KTILE = K_TILE / KB;
constexpr int HEAD_DIM = 128;
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * 2;
constexpr int Q_SUBTILES = 2;
constexpr int K_SUBTILES = 2;
constexpr int Q_SUB_BYTES = M_TILE * SUB_COLS_BYTES;  // 16 KB
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_BYTES;
constexpr int K_SUB_BYTES = K_TILE * SUB_COLS_BYTES;  // 16 KB
constexpr int KV_RING_SLOT_BYTES = 32 * 1024;         // one K-tile of K or of V
constexpr int NUM_KV_STAGES = 3;
constexpr int S_COLS = 128;
constexpr int K_ATOMS_PER_SUB = SUB_COLS_BF16 / 16;
constexpr int SPLIT_P_N = S_COLS / 4 * 3;
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;
constexpr int SPLIT_P_COL = SPLIT_P_N / 2;
constexpr int EX2_FRG_PAIRS = 16;
constexpr int EX2_FRG_CNT = S_COLS / 32;
constexpr int EX2_FREQ = 16;
constexpr int EX2_RES = 0;  // GB300 doubles the SFU ex2 rate: hardware ex2 for every pair is 1.5-5% faster than emulating 4 of 16
constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;
constexpr int STATS = M_TILE;
constexpr int STAT_REGIONS = 2;
constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;
constexpr int RESCALE_THRESHOLD = 8;
constexpr float P_SHIFT = 6.f;   // PV8: P <= 2^(P_SHIFT + RESCALE_LOG2) stays below E4M3's 448
constexpr int RESCALE_LOG2 = 2;

extern __shared__ __align__(1024) uint8_t fmha_pair_smem[];

union SmemDescPairT {
  uint64_t u64;
  uint2 w;
};

struct WorkItem {
  int sample, head, mtile0, mtile1, global_mtile0, global_mtile1, num_kv_blocks;
  int num_kv_blocks_mt[2];
};

__device__ __forceinline__ WorkItem decode_workitem(int workitem_id, int num_heads, int num_mtiles,
                                                    int packed_mtiles_per_seq, unsigned long long magic0,
                                                    unsigned long long magic2, const int* u_num) {
  WorkItem it;
  const int per_sample = num_heads * packed_mtiles_per_seq;
  it.sample = (int)fdiv((unsigned)workitem_id, magic0);
  const int rem = workitem_id - it.sample * per_sample;
  it.head = (int)fdiv((unsigned)rem, magic2);
  const int p = rem - it.head * packed_mtiles_per_seq;
  it.mtile0 = 2 * p;
  it.mtile1 = 2 * p + 1;
  it.global_mtile0 = (it.sample * num_heads + it.head) * num_mtiles + it.mtile0;
  it.global_mtile1 = it.global_mtile0 + 1;
  it.num_kv_blocks_mt[0] = u_num[it.global_mtile0];
  it.num_kv_blocks_mt[1] = u_num[it.global_mtile1];
  it.num_kv_blocks = max(max(it.num_kv_blocks_mt[0], it.num_kv_blocks_mt[1]), 1);
  return it;
}

__device__ __forceinline__ bool next_workitem(int& workitem_id, uint64_t* clc_full, uint64_t* clc_empty,
                                              uint32_t* clc_response, int& clc_stage, uint32_t& clc_phase) {
  ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(clc_full, clc_empty, clc_response,
                                                                                 clc_stage, clc_phase, elect_one_sync());
  clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
  if (!next.valid) return false;
  workitem_id = next.n_tile;
  return true;
}

__device__ __forceinline__ void mma_f8_ts(uint32_t lead, uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
                                          bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "@q tcgen05.mma.cta_group::1.kind::f8f6f4 [%1], [%2], %3, %4, {%6, %7, %8, %9}, p;\n\t"
      "}\n" ::"r"(lead),
      "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
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

__device__ __forceinline__ void mma_f8_ss(uint32_t lead, uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                          bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "@q tcgen05.mma.cta_group::1.kind::f8f6f4 [%1], %2, %3, %4, {%6, %7, %8, %9}, p;\n\t"
      "}\n" ::"r"(lead),
      "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}

// QK8 (needs PV8): Q and K arrive as E4M3 with static per-layer scales folded into scale_log2; QK^T runs as kind::f8f6f4.
template <bool PV8, bool QK8>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_pair_kernel(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_kv64,
                 const __grid_constant__ CUtensorMap tmap_v64, const __grid_constant__ CUtensorMap tmap_o, int seqlen,
                 int num_heads, int num_samples, int num_mtiles, int packed_mtiles_per_seq, int max_u,
                 unsigned long long magic0, unsigned long long magic2, const int* __restrict__ u_idx,
                 const uint8_t* __restrict__ u_mem, const int* __restrict__ u_num,
                 const int* __restrict__ variable_block_sizes, float scale_log2, const float* __restrict__ v_scale_blk,
                 const float* __restrict__ v_scale_head, int num_kblocks, float* __restrict__ lse_out) {
#if !defined(__CUDA_ARCH__) || ((__CUDA_ARCH__ == 1000 && defined(__CUDA_ARCH_FEAT_SM100_ALL)) || \
                                (__CUDA_ARCH__ == 1030 && defined(__CUDA_ARCH_FEAT_SM103_ALL)))
  uint8_t* sQ0 = fmha_pair_smem;
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
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_mtiles, packed_mtiles_per_seq, magic0, magic2, u_num);
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
          kv_block_id_cache[mtile_idx] = (cnt > 0) ? u_idx[(size_t)global_mtile[mtile_idx] * max_u + idx] : 0;
        }
        return __shfl_sync(0xffffffffu, kv_block_id_cache[mtile_idx], j & 31);
      };
      // One K-tile of K (or V): two 64-key blocks x two 64-column subtiles, laid out [sub][128 keys][128 B] like the
      // 128-token kernel's single 128-row box.
      auto load_kv = [&](int mtile_idx, int ktile_idx, const CUtensorMap* map) {
        const int kv_stage = kv_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        int blk_id[BLOCKS_PER_KTILE];
#pragma unroll
        for (int b = 0; b < BLOCKS_PER_KTILE; ++b) blk_id[b] = get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + b);
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), KV_RING_SLOT_BYTES);
#pragma unroll
          for (int b = 0; b < BLOCKS_PER_KTILE; ++b)
#pragma unroll
            for (int sub = 0; sub < K_SUBTILES; ++sub)
              tma_load_4d(smem_ptr_u32(sKV + kv_stage * KV_RING_SLOT_BYTES + sub * K_SUB_BYTES + b * KB * SUB_COLS_BYTES),
                          map, smem_ptr_u32(&full_bar[kv_stage]), 0, blk_id[b] * KB, sub, bh);
        }
      };
      auto load_v8 = [&](int mtile_idx, int ktile_idx) {
        const int kv_stage = kv_empty_ph.get_stage();
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        int blk_id[BLOCKS_PER_KTILE];
#pragma unroll
        for (int b = 0; b < BLOCKS_PER_KTILE; ++b) blk_id[b] = get_kv_block_id(mtile_idx, ktile_idx * BLOCKS_PER_KTILE + b);
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), BLOCKS_PER_KTILE * KB * HEAD_DIM);
#pragma unroll
          for (int b = 0; b < BLOCKS_PER_KTILE; ++b)
            tma_load_3d(smem_ptr_u32(sKV + kv_stage * KV_RING_SLOT_BYTES + b * KB * HEAD_DIM), &tmap_v64,
                        smem_ptr_u32(&full_bar[kv_stage]), 0, blk_id[b] * KB, bh);
        }
      };
      auto load_v = [&](int m, int kt) {
        if constexpr (PV8) load_v8(m, kt);
        else load_kv(m, kt, &tmap_v64);
      };
      // E4M3 K: same shape as E4M3 V (one 64-row x 128-byte box per block, keys contiguous = K-major rows).
      auto load_k = [&](int m, int kt) {
        if constexpr (QK8) {
          const int kv_stage = kv_empty_ph.get_stage();
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
          kv_empty_ph.advance();
          int blk_id[BLOCKS_PER_KTILE];
#pragma unroll
          for (int b = 0; b < BLOCKS_PER_KTILE; ++b) blk_id[b] = get_kv_block_id(m, kt * BLOCKS_PER_KTILE + b);
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), BLOCKS_PER_KTILE * KB * HEAD_DIM);
#pragma unroll
            for (int b = 0; b < BLOCKS_PER_KTILE; ++b)
              tma_load_3d(smem_ptr_u32(sKV + kv_stage * KV_RING_SLOT_BYTES + b * KB * HEAD_DIM), &tmap_kv64,
                          smem_ptr_u32(&full_bar[kv_stage]), 0, blk_id[b] * KB, bh);
          }
        } else {
          load_kv(m, kt, &tmap_kv64);
        }
      };
      load_k(0, 0);
      load_k(1, 0);
#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        if (elect_one_sync()) {
          if constexpr (QK8) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), M_TILE * HEAD_DIM);
            tma_load_3d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]), 0, mtile[m] * M_TILE, bh);
          } else {
            mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), Q_TILE_BYTES);
            tma_load_4d(smem_ptr_u32(sQ[m]), &tmap_q, smem_ptr_u32(&full_bar_q[m]), 0, bh, mtile[m] * M_TILE, 0);
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
      if (!next_workitem(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  } else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();
    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    const uint32_t idesc_qk = QK8 ? make_idesc_table44(M_TILE, K_TILE, 1, 0, 0, false, false)
                                  : make_idesc_bf16_f32(M_TILE, K_TILE, false, false);
    const uint32_t idesc_pv = PV8 ? make_idesc_table44(M_TILE, HEAD_DIM, 1, 0, 0, false, true)
                                  : make_idesc_bf16_f32(M_TILE, HEAD_DIM, false, true);
    // E4M3 V: one 128-byte row per key, N = 128 dims = one swizzle atom wide; K = 32 keys (4 KB) per MMA.
    const uint64_t desc_v8 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, 8192, SmemSwizzleBlackwell::B128);
    constexpr uint32_t DESC_LBO_MN = (uint32_t)(K_TILE * SUB_COLS_BYTES);
    const uint64_t desc_v0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO_MN, SmemSwizzleBlackwell::B128);
    constexpr uint32_t PV_DESC_STEP = (uint32_t)((16 * SUB_COLS_BYTES) >> 4);
    const uint64_t desc_q0 = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    constexpr uint64_t KV_DESC_DELTA = KV_RING_SLOT_BYTES >> 4;
    constexpr uint64_t Q_SUB_DELTA = Q_SUB_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;
    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_mtiles, packed_mtiles_per_seq, magic0, magic2, u_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;
      auto bmm1 = [&](int i) {
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint64_t da_base = desc_q0 + (uint64_t)i * Q_MTILE_DESC_DELTA;
        const int slot = kv_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
        kv_ph.advance();
        if constexpr (QK8) {
          const uint64_t db = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA;
#pragma unroll
          for (int ki = 0; ki < HEAD_DIM / 32; ++ki)  // K = 32 E4M3 per MMA, 32 bytes = 2 descriptor units
            mma_f8_ss(lead, s_tmem_addr, da_base + 2 * ki, db + 2 * ki, idesc_qk, ki != 0);
        } else {
#pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            const uint64_t da = da_base + (uint64_t)s * Q_SUB_DELTA;
            const uint64_t db = desc_kv0 + (uint64_t)slot * KV_DESC_DELTA + (uint64_t)s * (K_SUB_BYTES >> 4);
#pragma unroll
            for (int ki = 0; ki < K_ATOMS_PER_SUB; ++ki)
              tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da + 2 * ki, db + 2 * ki, idesc_qk, (s != 0) || (ki != 0));
          }
        }
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[slot]));
        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
      };
      auto bmm2 = [&](int i, bool first_ktile, auto last_c) {
        constexpr bool last_ktile = decltype(last_c)::value;
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        const int slot = kv_ph.get_stage();
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[slot]), kv_ph.get_phase());
        kv_ph.advance();
        if constexpr (PV8) {
          const uint64_t dv = desc_v8 + (uint64_t)slot * KV_DESC_DELTA;
#pragma unroll
          for (int a = 0; a < K_TILE / 32; ++a) {
            if (a == 2) mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            mma_f8_ts(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dv + (uint64_t)((a * 32 * HEAD_DIM) >> 4),
                      idesc_pv, (!first_ktile) || (a != 0));
          }
        } else {
          SmemDescPairT desc_bv;
          desc_bv.u64 = desc_v0;
          desc_bv.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA);
#pragma unroll
          for (int a = 0; a < K_TILE / 16; ++a) {
            if (a == SPLIT_P_ATOM) mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
            tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), desc_bv.u64, idesc_pv,
                                        (!first_ktile) || (a != 0));
            desc_bv.w.x += PV_DESC_STEP;
          }
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
      if (!next_workitem(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
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
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_mtiles, packed_mtiles_per_seq, magic0, magic2, u_num);
      const int mtile[2] = {it.mtile0, it.mtile1};
#pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        if (elect_one_sync()) {
          tma_store_4d(&tmap_o, 0, it.sample * num_heads + it.head, mtile[m] * M_TILE, 0,
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
      if (!next_workitem(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
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
    const int corr_row = corr_tid;
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
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_mtiles, packed_mtiles_per_seq, magic0, magic2, u_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;
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
        const float l = alpha_and_l_smem[(i * STAT_REGIONS + 1) * STATS + corr_tid];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
        const float vs_head = PV8 ? v_scale_head[it.sample * num_heads + it.head] : 1.f;
        const float scale_own = (l > 0.f) ? rcp_approx_ftz_f32(l) * vs_head : 0.f;
        const float2 scale2 = f32x2_splat(scale_own);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
#pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o_regs[16];
          tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
          if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          const float2* o2 = reinterpret_cast<const float2*>(o_regs);
          const int s = c0 / SUB_COLS_BF16;
          const int v_base = (c0 % SUB_COLS_BF16) / 8;
          __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
#pragma unroll
          for (int vv = 0; vv < 2; ++vv) {
            const int v = v_base + vv;
            const float2 r0 = fmul2(o2[vv * 4 + 0], scale2);
            const float2 r1 = fmul2(o2[vv * 4 + 1], scale2);
            const float2 r2 = fmul2(o2[vv * 4 + 2], scale2);
            const float2 r3 = fmul2(o2[vv * 4 + 3], scale2);
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
            packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
            packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
            packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
            *reinterpret_cast<uint4*>(&so_sub[corr_row * SUB_COLS_BF16 + (v ^ (corr_row & 7)) * 8]) = packed;
          }
        }
        tcgen05_fence_before_thread_sync();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
        fence_proxy_async_shared();
        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();
      if (!next_workitem(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  } else {
    setmaxnreg_inc<192>();
    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    const int sm_tid = warp_in_group * 32 + lane;
    // Rows 0-63 of the 128-row tile are the pair's first 64-token tile, rows 64-127 the second.
    const uint32_t mem_bit = warp_in_group < 2 ? 1u : 2u;
    const uint32_t alpha_slot_u32 = smem_ptr_u32(&alpha_and_l_smem[(m_tile * STAT_REGIONS + 0) * STATS + sm_tid]);
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    int clc_stage = 0;
    uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      const WorkItem it = decode_workitem(workitem_id, num_heads, num_mtiles, packed_mtiles_per_seq, magic0, magic2, u_num);
      const int num_k_tiles = (it.num_kv_blocks + BLOCKS_PER_KTILE - 1) / BLOCKS_PER_KTILE;
      const int gqt = (m_tile == 0) ? it.global_mtile0 : it.global_mtile1;
      const int cnt = it.num_kv_blocks_mt[m_tile];
      float m_run = -INFINITY, l_run = 0.f;
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      // Per 32-K-tile window each lane caches its K-tile's two thresholds: a block this row's tile did not select
      // masks completely (threshold 0); a selected one masks its pad keys (vbs).
      int thr_window = -(1 << 30);
      int thr_cache0 = KB, thr_cache1 = KB;
      float lr_cache0 = 0.f, lr_cache1 = 0.f;
      const int bh = it.sample * num_heads + it.head;
      const float lv_head = PV8 ? __log2f(v_scale_head[bh]) : 0.f;
      auto get_thresholds = [&](int k, int& t0, int& t1, float& lr0, float& lr1) {
        if (k >= thr_window + 32) {
          thr_window = k & ~31;
          const int b0 = (thr_window + lane) * BLOCKS_PER_KTILE;
          const size_t base = (size_t)gqt * max_u;
          const int id0 = b0 < cnt ? u_idx[base + b0] : -1, id1 = b0 + 1 < cnt ? u_idx[base + b0 + 1] : -1;
          thr_cache0 = (id0 >= 0 && (u_mem[base + b0] & mem_bit)) ? variable_block_sizes[id0] : 0;
          thr_cache1 = (id1 >= 0 && (u_mem[base + b0 + 1] & mem_bit)) ? variable_block_sizes[id1] : 0;
          if constexpr (PV8) {
            const float* vs_row = v_scale_blk + (size_t)bh * num_kblocks;
            lr_cache0 = id0 >= 0 ? __log2f(fmaxf(vs_row[id0], 1e-30f)) - lv_head : 0.f;
            lr_cache1 = id1 >= 0 ? __log2f(fmaxf(vs_row[id1], 1e-30f)) - lv_head : 0.f;
          }
        }
        t0 = __shfl_sync(0xffffffffu, thr_cache0, k & 31);
        t1 = __shfl_sync(0xffffffffu, thr_cache1, k & 31);
        lr0 = __shfl_sync(0xffffffffu, lr_cache0, k & 31);
        lr1 = __shfl_sync(0xffffffffu, lr_cache1, k & 31);
      };
      auto softmax_step = [&](auto is_first_c, int k) {
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        int thr0, thr1;
        float lr0, lr1;
        get_thresholds(k, thr0, thr1, lr0, lr1);
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
        if (thr0 < KB) {
          mask_s_row_r2p<false, 64>(scores, 0, 0, thr0);
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
        if (thr1 < KB) {
          mask_s_row_r2p<false, 64>(scores + 64, 0, 0, thr1);
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
          const float acc_scale = (m_run - new_m) * scale_log2;
          if (acc_scale >= -(float)(PV8 ? RESCALE_LOG2 : RESCALE_THRESHOLD)) {
            new_m = m_run;
            alpha = 1.0f;
          } else {
            alpha = ex2_approx_f32(acc_scale);
          }
          sts_f32(alpha_slot_u32, alpha);
        }
        mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));
        const float2 scale2 = f32x2_splat(scale_log2);
        float2 lt2 = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f);
        if constexpr (PV8) {
          const float base = -new_m * scale_log2 + P_SHIFT;
          const float2 nm0 = f32x2_splat(base + lr0), nm1 = f32x2_splat(base + lr1);
          float2 la = make_float2(0.f, 0.f), lb = make_float2(0.f, 0.f);
#pragma unroll
          for (int c = 0; c < S_COLS / 2; ++c) {
            const float2 a2 = ffma2(scores2[c], scale2, c < S_COLS / 4 ? nm0 : nm1);
            const int jj = c / EX2_FRG_PAIRS;
            const int kk = 2 * (c % EX2_FRG_PAIRS);
            const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1);
            scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y)) : ex2_emu_f32x2(a2.x, a2.y);
            if (c < S_COLS / 4) la = fadd2(la, scores2[c]);
            else lb = fadd2(lb, scores2[c]);
          }
          uint32_t p8[S_COLS / 4];
#pragma unroll
          for (int c = 0; c < S_COLS / 4; ++c)
            p8[c] = e4m3x4(scores2[2 * c].x, scores2[2 * c].y, scores2[2 * c + 1].x, scores2[2 * c + 1].y);
          tcgen05_st_32x32b_x16(s_tmem_addr, *reinterpret_cast<uint32_t(*)[16]>(&p8[0]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
          tcgen05_st_32x32b_x16(s_tmem_addr + 16, *reinterpret_cast<uint32_t(*)[16]>(&p8[16]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
          lt2.x += (la.x + la.y) * ex2_approx_f32(-lr0) + (lb.x + lb.y) * ex2_approx_f32(-lr1);
        } else {
          const float2 neg_m_scaled2 = f32x2_splat(-new_m * scale_log2);
          uint32_t p_regs[S_COLS / 2];
#pragma unroll
          for (int c = 0; c < S_COLS / 2; ++c) {
            const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
            const int jj = c / EX2_FRG_PAIRS;
            const int kk = 2 * (c % EX2_FRG_PAIRS);
            const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1);
            scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y)) : ex2_emu_f32x2(a2.x, a2.y);
            lt2 = fadd2(lt2, scores2[c]);
            p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
          }
          tcgen05_st_32x32b_x32(s_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x16(s_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
          tcgen05_st_32x32b_x16(s_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        }
        spo_ph.advance();
        l_run = lt2.x + lt2.y;
        m_run = new_m;
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
      };
      if (num_k_tiles > 0) softmax_step(std::true_type{}, 0);
      for (int k = 1; k < num_k_tiles; ++k) softmax_step(std::false_type{}, k);
      alpha_and_l_smem[(m_tile * STAT_REGIONS + 1) * STATS + sm_tid] = l_run;
      if (lse_out != nullptr) {
        // log2-domain LSE of this row's scores (l_run carries the PV8 2^P_SHIFT; the V-block factors are divided out)
        const int q_row = (m_tile == 0 ? it.mtile0 : it.mtile1) * M_TILE + sm_tid;
        lse_out[(size_t)(it.sample * num_heads + it.head) * seqlen + q_row] =
            l_run > 0.f ? m_run * scale_log2 + __log2f(l_run) - (PV8 ? P_SHIFT : 0.f) : -INFINITY;
      }
      mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
      if (!next_workitem(workitem_id, clc_full, clc_empty, clc_response, clc_stage, clc_phase)) break;
    }
  }
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
#endif
}

struct Args {
  const __nv_bfloat16* q;  // [B, H, S, 128], S = num_mtiles * 128
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  __nv_bfloat16* o;
  const int* u_idx;        // [B*H*num_mtiles, max_u]: union of the two 64-token tiles' sorted lists
  const uint8_t* u_mem;    // same shape: bit 0 / bit 1 = first / second tile selected the block
  const int* u_num;        // [B*H*num_mtiles]
  const int* variable_block_sizes;  // [S / 64]
  int batch, num_heads, seqlen, max_u;
  float sm_scale;
  const uint8_t* v8;                // PV8: E4M3 V [B*H, S, 128]; nullptr runs PV in BF16
  bool qk8;                         // q and k point at E4M3 [B*H, S, 128]; sm_scale includes their static scales
  const float* v_scale_blk;         // PV8: [B*H, S / 64]
  const float* v_scale_head;        // PV8: [B*H]
  float* lse;                       // optional [B*H, S] log2-domain LSE
};

inline cudaError_t launch(const Args& a, cudaStream_t stream) {
  const int B = a.batch, H = a.num_heads, S = a.seqlen;
  if (S % (2 * M_TILE) != 0 || a.max_u < 1) return cudaErrorInvalidValue;
  const int num_mtiles = S / M_TILE;
  const int packed_mtiles_per_seq = num_mtiles / 2;
  const int total_work = B * H * packed_mtiles_per_seq;
  CUtensorMap tq, tk, tv, to;
  {
    // Q/O as the BF16 kernel's BHSD map: [64 cols][B*H][S][2 subtiles], box 64 x 1 x 128 rows x 2.
    uint64_t gd[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)((long)B * H), (uint64_t)S, (uint64_t)Q_SUBTILES};
    uint64_t gs[3] = {(uint64_t)((long)S * HEAD_DIM) * 2u, (uint64_t)HEAD_DIM * 2u, (uint64_t)SUB_COLS_BF16 * 2u};
    uint32_t bd[4] = {(uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, (uint32_t)Q_SUBTILES};
    uint32_t es[4] = {1u, 1u, 1u, 1u};
    if ((!a.qk8 &&
         cuTensorMapEncodeTiled(&tq, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(a.q), gd, gs, bd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) ||
        cuTensorMapEncodeTiled(&to, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, a.o, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  {
    // K and V: [64 cols][S][2 subtiles][B*H], box 64 cols x 64 keys: one key block of one subtile per load.
    uint64_t gd[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)S, (uint64_t)K_SUBTILES, (uint64_t)((long)B * H)};
    uint64_t gs[3] = {(uint64_t)HEAD_DIM * 2u, (uint64_t)SUB_COLS_BF16 * 2u, (uint64_t)((long)S * HEAD_DIM) * 2u};
    uint32_t bd[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)KB, 1u, 1u};
    uint32_t es[4] = {1u, 1u, 1u, 1u};
    if ((!a.qk8 &&
         cuTensorMapEncodeTiled(&tk, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(a.k), gd, gs, bd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) ||
        (a.v8 == nullptr &&
         cuTensorMapEncodeTiled(&tv, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(a.v), gd, gs, bd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS))
      return cudaErrorInvalidValue;
  }
  auto fp8_map = [&](CUtensorMap* map, const void* base, uint32_t rows) {
    uint64_t gd[3] = {(uint64_t)HEAD_DIM, (uint64_t)S, (uint64_t)((long)B * H)};
    uint64_t gs[2] = {(uint64_t)HEAD_DIM, (uint64_t)((long)S * HEAD_DIM)};
    uint32_t bd[3] = {(uint32_t)HEAD_DIM, rows, 1u};
    uint32_t es[3] = {1u, 1u, 1u};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), gd, gs, bd, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
  };
  if (a.v8 != nullptr && !fp8_map(&tv, a.v8, KB)) return cudaErrorInvalidValue;
  if (a.qk8 && (a.v8 == nullptr || !fp8_map(&tq, a.q, M_TILE) || !fp8_map(&tk, a.k, KB))) return cudaErrorInvalidValue;
  const size_t smem = (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_RING_SLOT_BYTES +
                      (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16) + (2 * NUM_KV_STAGES + 22) * 8 +
                      (size_t)CLC_STAGES * (2 * 8 + 16) + 16 + 8 + (size_t)2 * STAT_REGIONS * STATS * sizeof(float) + 256;
  auto kfn = a.qk8 ? fmha_pair_kernel<true, true>
                   : (a.v8 != nullptr ? fmha_pair_kernel<true, false> : fmha_pair_kernel<false, false>);
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
  return cudaLaunchKernelEx(&cfg, kfn, tq, tk, tv, to, S, H, B, num_mtiles, packed_mtiles_per_seq, a.max_u, magic0, magic2,
                            a.u_idx, a.u_mem, a.u_num, a.variable_block_sizes, a.sm_scale * 1.4426950408889634f,
                            a.v_scale_blk, a.v_scale_head, S / KB, a.lse);
}

}  // namespace vsa_pair

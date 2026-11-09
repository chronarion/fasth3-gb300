# SPDX-License-Identifier: Apache-2.0
"""Fused single-GPU inference path for the MiniMax-H3 DiT blocks (FP8 or NVFP4 linears, block-sparse VSA attention).

The block loop runs on compact tile-order rows: one permute into tile order per forward, q/k/v written straight into the
attention kernel's heads-first padded layout, top-k written straight into its index list, FP8 row scales folded into the
consumer of every GEMM. See ``h3fast.cu``. Attention: the stock ptx_blackwell kernels (BF16), ``vsa_fp8.cuh`` (64-token
tiles, E4M3 PV or all-8-bit), ``vsa_pair.cuh`` (128-token tiles), and for 128-token layouts with an exempt prefix the
dense prefix on FlashAttention-4 merged with the sparse rest by log-sum-exp (FASTVIDEO_H3_DENSE_PREFIX).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import fastvideo.envs as envs
from fastvideo.attention.backends.video_sparse_attn import compute_topk
from fastvideo.layers.quantization.nvfp4_config import _mm_fp4_backend, _require_flashinfer
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_ext: Any = None


def ext() -> Any:
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        build = Path(os.environ.get("FASTVIDEO_H3_FAST_BUILD", "/cache/h3fast"))
        build.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).parent
        _ext = load(name="h3fast",
                    sources=[str(here / f) for f in ("h3fast.cu", "h3attn8.cu", "h3attn16.cu", "h3attn16b.cu")],
                    build_directory=str(build),
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_103a,code=sm_103a",
                                       "-std=c++17"],
                    extra_ldflags=["-lcuda"],
                    verbose=False)
    return _ext


def _vsa_forward(tile: int) -> Any:
    """BF16 sparse attention: h3fast's copy of FastVideo's kernel (vsa_bf16.cuh: row max from tcgen05.ld.red, every exp
    on the hardware ex2) or, with FASTVIDEO_H3_BF16_VSA=stock, FastVideo's own op. Same arguments and results."""
    if os.environ.get("FASTVIDEO_H3_BF16_VSA", "h3") != "stock":
        return ext().vsa_bf16 if tile == 64 else ext().vsa_bf16_blk128
    from fastvideo_kernel._C import fastvideo_kernel_ops as ops
    return ops.block_sparse_ptx_blackwell_fwd if tile == 64 else ops.block_sparse_ptx_blackwell_blk128_fwd


def _base(layer: Any) -> Any:
    """The linear under a FastVideo LoRA wrapper (``BaseLayerWithLoRA.base_layer``), or the layer itself."""
    return getattr(layer, "base_layer", layer)


def _adapter(layer: Any) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """(A, B, scale) of the side-branch LoRA adapter that ``layer`` adds to its base output (``y = base(x) + scale *
    B(A x)``, scale = strength * alpha / rank), or None: no wrapper, no adapter for this layer, or strength 0. A
    merged adapter is in the base weight already."""
    if getattr(layer, "lora_A", None) is None or not getattr(layer, "side_branch", False):
        return None
    if layer._strength_is_zero or layer.disable_lora or layer.merged:
        return None
    rank, alpha = layer.lora_rank, layer.lora_alpha
    scale = layer.lora_strength * (float(alpha) / float(rank) if alpha != rank else 1.0)
    return layer.slice_lora_a_weights(layer.lora_A), layer.slice_lora_b_weights(layer.lora_B), scale


def _gate_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate-compress weight in E4M3 with one float32 scale (the gate linear stays BF16 in the model)."""
    scale = (weight.abs().amax().float() / _FP8_MAX).clamp(min=1e-12)
    return (weight.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8), scale


def _on_grid(merged: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``merged`` (fp32, a LoRA folded into a weight) in E4M3 on the base weight's grid ``scale`` (one per tensor or
    per output row): the same rounding grid keeps the merged model's quantization error that of the base (a fresh
    scale measured +1-3% teacher-forced error). A tensor or row that outgrows the grid's range by more than 5% gets
    its own scale instead of clipping."""
    rows = scale.numel() > 1
    amax = merged.abs().amax(dim=1) if rows else merged.abs().amax().reshape(1)
    scale = torch.where(amax <= 1.05 * _FP8_MAX * scale.reshape(-1), scale.reshape(-1),
                        (amax / _FP8_MAX).clamp(min=1e-12)).float()
    q = (merged / (scale[:, None] if rows else scale)).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return q, scale if rows else scale.reshape(1)


@dataclass
class _Layout:
    key: tuple
    rows: int
    tile: int
    n_tiles: int
    n_ext: int
    perm: torch.Tensor  # compact row -> packed row
    inv: torch.Tensor  # packed row -> compact row
    slot: torch.Tensor  # compact row -> padded slot
    tile_row0: torch.Tensor
    vbs: torch.Tensor
    vbs_ext: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    adaln: torch.Tensor
    buffers: dict[str, torch.Tensor]
    n_pre: int = 0  # compact rows of the prefix tiles when the dense prefix passes are on


class H3FastRunner:

    def __init__(self, model: Any) -> None:
        self.model = model
        self.blocks = list(model.transformer_blocks)
        block = self.blocks[0]
        self.heads = block.attn.num_attention_heads
        self.head_dim = block.attn.attention_head_dim
        self.hidden = block.norm1.weight.numel()
        fc_out = _base(block.ff.fc_out)
        self.ffn = fc_out.input_size if hasattr(fc_out, "input_size") else 14336
        self.fp4 = getattr(_base(block.attn.to_q), "_nvfp4_weight", None) is not None
        # "pv": BF16 QK^T + E4M3 PV; "1": E4M3 QK^T too (faster, larger error).
        mode = os.environ.get("FASTVIDEO_H3_FP8_ATTN", "0")
        self.fp8_attn = mode in ("1", "pv")
        self.qk16 = mode == "pv"
        # 128-token tiles (Omni) with FP8 attention run on the M=128 pair kernel (E4M3 PV); BF16 stays on the stock
        # blk128 kernel, which is faster than the pair kernel's BF16 build.
        self.pair128 = self.fp8_attn
        # Layouts with an exempt prefix (text, audio, references): the prefix queries (against every key) and the prefix
        # keys (for every other query) as two dense FA4 calls, BF16 or E4M3 (Q/K static scales, V one scale per head);
        # the sparse kernel keeps only the selected video tiles and returns its LSE, and an LSE merge in the gate kernel
        # joins the two (every sparse route returns an LSE: the stock BF16 kernels, vsa_fp8 and the pair kernel). On
        # when the prefix has at least FASTVIDEO_H3_DENSE_PREFIX_MIN_ROWS rows (long prompts, references).
        self.dense_prefix = os.environ.get("FASTVIDEO_H3_DENSE_PREFIX", "0")
        if self.dense_prefix not in ("0", "bf16", "fp8"):
            raise ValueError(f"FASTVIDEO_H3_DENSE_PREFIX={self.dense_prefix!r}: expected 0, bf16 or fp8")
        if self.dense_prefix == "fp8" and not self.fp8_attn:
            logger.warning("FASTVIDEO_H3_DENSE_PREFIX=fp8 needs FASTVIDEO_H3_FP8_ATTN (pv or 1); using bf16")
            self.dense_prefix = "bf16"
        self.dense_prefix_min_rows = int(os.environ.get("FASTVIDEO_H3_DENSE_PREFIX_MIN_ROWS", "0"))
        # TF32 for the two small fp32 matmuls on tile means (scores for the top-k, compressed branch): 0.29 -> 0.06 ms
        # per block. Selection can move at near-ties (the stock path scores in fp32); the teacher-forced error vs BF16
        # did not go up (0.194 / 0.071 -> 0.191 / 0.068). "0" keeps fp32.
        self.tf32_scores = os.environ.get("FASTVIDEO_H3_TF32_SCORES", "1") != "0"
        # The gate-compress GEMM and the compressed branch on a second stream: both only feed attn_gate_quant, so they
        # can overlap the prologue / scores / top-k and the sparse attention.
        self.side = torch.cuda.Stream() if os.environ.get("FASTVIDEO_H3_SIDE_STREAM", "1") == "1" else None
        self.layouts: dict[tuple, _Layout] = {}
        self.one: torch.Tensor | None = None
        self._gate_fp8: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lora_dev: dict[int, tuple[Any, torch.Tensor, torch.Tensor]] = {}
        # FASTVIDEO_H3_LORA_MERGE: FP8 weights with the active LoRA adapter merged in, per LoRA layer
        self.lora_merge = envs.FASTVIDEO_H3_LORA_MERGE.get()
        self._merged: dict[int, tuple[tuple, torch.Tensor, torch.Tensor]] = {}
        # (input, A) -> x A^T of the last side branch; reset wherever the input buffers get new contents
        self._lora_t: tuple[tuple, torch.Tensor] | None = None
        ext()

    @staticmethod
    def supported(model: Any, meta: Any) -> bool:
        block = model.transformer_blocks[0]
        attn = block.attn
        fp8 = getattr(_base(attn.to_q), "_fp8_weight", None) is not None
        fp4 = all(getattr(_base(lin), "_nvfp4_weight", None) is not None
                  for lin in (attn.to_q, attn.to_k, attn.to_v, attn.to_out, block.ff.fc_in, block.ff.fc_out))
        return (fp8 or fp4) and meta is not None and meta.exempt and not meta.pad_tokens and meta.tile_elems in (64, 128)

    def _layer_gate(self, index: int, attn: Any) -> Any:
        if attn.to_gate_compress is None or not attn._gate_active():
            return None
        if self.fp4:
            return attn.to_gate_compress
        entry = self._gate_fp8.get(index)
        if entry is None:
            entry = self._gate_fp8[index] = _gate_quant(_base(attn.to_gate_compress).weight.data)
        return entry

    def _get_layout(self, rows: int, device: torch.device, meta: Any, rotary: tuple[torch.Tensor, torch.Tensor],
                    adaln: torch.Tensor, fresh: bool = True) -> _Layout:
        """The layout (row maps, tables, buffers) for this metadata, cached across steps. A new full layout (``fresh``)
        drops every other one first; the token-reduction layout (``fresh=False``) lives beside the full one."""
        key = (meta.untile_combined_index.data_ptr(), rows, meta.tile_elems, rotary[0].data_ptr())
        layout = self.layouts.get(key)
        if layout is not None:
            layout.adaln = adaln.index_select(0, layout.perm).to(torch.int32)
            return layout
        if fresh:
            self.layouts.clear()
            self.__dict__.pop("_pair_cache", None)
        tile = int(meta.tile_elems)
        untile = meta.untile_combined_index
        perm = torch.argsort(untile)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(rows, device=device)
        vbs = meta.variable_block_sizes.to(torch.int32)
        n_tiles = vbs.numel()
        n_ext = n_tiles + n_tiles % 2
        vbs_ext = torch.nn.functional.pad(vbs, (0, n_ext - n_tiles))
        tile_row0 = (vbs.cumsum(0) - vbs).to(torch.int32)
        heads, dim, hidden = self.heads, self.head_dim, self.hidden
        rows_pad = n_ext * tile
        e = lambda *s, d=torch.bfloat16: torch.empty(*s, device=device, dtype=d)  # noqa: E731
        buffers = {
            "x": e(rows, hidden), "q": e(rows, heads * dim), "k": e(rows, heads * dim), "v": e(rows, heads * dim),
            "g": e(rows, heads * dim), "y": e(rows, hidden), "u": e(rows, 2 * self.ffn),
            "hs": e(rows, d=torch.float32), "as": e(rows, d=torch.float32), "fs": e(rows, d=torch.float32),
            "qp": e(heads, n_tiles, dim, d=torch.float32), "kp": e(heads, n_tiles, dim, d=torch.float32),
            "vp": e(heads, n_tiles, dim, d=torch.float32), "sc": e(heads, n_tiles, n_tiles, d=torch.float32),
            "idx": torch.zeros(1, heads, n_ext, n_tiles, device=device, dtype=torch.int32),
            "num": torch.zeros(1, heads, n_ext, device=device, dtype=torch.int32),
        }
        pad = (rows + 127) // 128 * 128
        for name, width in (("h", hidden), ("a", heads * dim), ("f", self.ffn)):
            if self.fp4:
                buffers[name] = e(rows, width // 2, d=torch.uint8)
                buffers[name + "_sf"] = torch.zeros(pad, width // 16, device=device, dtype=torch.uint8)
            else:
                buffers[name] = e(rows, width, d=_FP8)
        # Pad rows stay zero: nothing writes them, and the kernel masks them through vbs.
        qk16 = self.qk16
        for name in ("qb", "kb", "vb"):
            eight = self.fp8_attn and (name == "vb" or not qk16)
            buffers[name] = torch.zeros(1, heads, rows_pad, dim, device=device,
                                        dtype=torch.uint8 if eight else torch.bfloat16)
        if self.fp8_attn:
            buffers["vs_blk"] = torch.ones(heads, n_ext * tile // 64, device=device, dtype=torch.float32)
            buffers["vs_head"] = torch.zeros(heads, device=device, dtype=torch.float32)
        if tile == 128 and self.pair128:
            buffers["uidx"] = torch.zeros(1, heads, n_ext, 2 * n_tiles, device=device, dtype=torch.int32)
            buffers["umem"] = torch.full((1, heads, n_ext, 2 * n_tiles), 3, device=device, dtype=torch.uint8)
            buffers["unum"] = torch.zeros(1, heads, n_ext, device=device, dtype=torch.int32)
            buffers["vbs64"] = torch.stack([vbs_ext.clamp(max=64), (vbs_ext - 64).clamp(min=0)], -1).flatten().contiguous()
        n_pre = int(vbs[:meta.num_prefix_tiles].sum().item()) if meta.num_prefix_tiles > 0 else 0
        if self.dense_prefix == "0" or n_pre == 0 or n_pre < self.dense_prefix_min_rows:
            n_pre = 0
        if n_pre:
            qk = torch.bfloat16 if self.dense_prefix == "bf16" else _FP8
            buffers["qc"] = e(rows, heads, dim, d=qk)
            buffers["kc"] = e(rows, heads, dim, d=qk)
            buffers["vc"] = e(rows, heads, dim)
            if self.dense_prefix == "fp8":
                buffers["vc8"] = e(rows, heads, dim, d=_FP8)
            buffers["o1"] = e(1, rows, heads, dim)
            buffers["lse1a"] = e(1, heads, n_pre, d=torch.float32)
            buffers["lse1b"] = e(1, heads, rows - n_pre, d=torch.float32)
            if self.fp8_attn:  # the 8-bit kernels write into it; the stock BF16 kernel returns its own
                buffers["lse2"] = e(1, heads, rows_pad, d=torch.float32)
        cos, sin = rotary
        layout = _Layout(key, rows, tile, n_tiles, n_ext, perm, inv, untile.index_select(0, perm).to(torch.int32),
                         tile_row0, vbs, vbs_ext, cos.index_select(0, perm).contiguous(),
                         sin.index_select(0, perm).contiguous(), adaln.index_select(0, perm).to(torch.int32), buffers,
                         n_pre)
        self.layouts[key] = layout
        self.one = torch.ones((), device=device, dtype=torch.float32)
        return layout

    def _pair_maps(self, plan: Any, L: _Layout, R: _Layout) -> dict[str, Any]:
        """Token reduction index maps between the compact rows of the full layout L and of the paired layout R."""
        key = (L.key, R.key, plan.first.data_ptr(), plan.parents.data_ptr())
        maps = self.__dict__.setdefault("_pair_cache", {}).get(key)
        if maps is None:
            self._pair_cache.clear()
            # reduced compact row -> the two full compact rows it averages (equal for unpaired rows)
            ia = L.inv.index_select(0, plan.first.index_select(0, R.perm))
            ib = L.inv.index_select(0, plan.second.index_select(0, R.perm))
            # full compact row -> its reduced compact row, and whether it belongs to a true pair
            par = R.inv.index_select(0, plan.parents.index_select(0, L.perm))
            in_pair = torch.zeros(L.rows, device=ia.device, dtype=torch.bool)
            in_pair[plan.pair_rows] = True
            maps = self._pair_cache[key] = {
                "ia": ia, "ib": ib, "par": par, "pair": in_pair.index_select(0, L.perm)[:, None],
                "pooled": torch.empty(R.rows, self.hidden, device=ia.device, dtype=torch.bfloat16)}
        return maps

    def _pool(self, L: _Layout, R: _Layout, maps: dict[str, Any]) -> None:
        """Paired rows into R's residual stream: the fp32 mean of both sources (unpaired rows copy exactly)."""
        xa = L.buffers["x"].index_select(0, maps["ia"]).float()
        xa.add_(L.buffers["x"].index_select(0, maps["ib"]).float()).mul_(0.5)
        R.buffers["x"].copy_(xa)
        maps["pooled"].copy_(R.buffers["x"])

    def _expand(self, L: _Layout, R: _Layout, maps: dict[str, Any], scale: float) -> None:
        """Full rows back from R: a paired row is its value before pooling plus scale x its pair's update; every
        other row (prefix, an odd-width video's last column) takes the reduced output directly."""
        xr = R.buffers["x"]
        update = (xr.float() - maps["pooled"].float()).index_select(0, maps["par"])
        paired = update.mul_(scale).add_(L.buffers["x"].float())
        L.buffers["x"].copy_(torch.where(maps["pair"], paired, xr.index_select(0, maps["par"]).float()))

    def _qk_scales(self, index: int, attn: Any) -> tuple[float, float]:
        cached = self.__dict__.setdefault("_qk_scale_cache", {}).get(index)
        if cached is None:
            bound = math.sqrt(self.head_dim) / _FP8_MAX
            s_q = bound * attn.norm_q.weight.float().abs().max().item()
            s_k = bound * attn.norm_k.weight.float().abs().max().item()
            cached = self._qk_scale_cache[index] = (s_q, s_k)
        return cached

    def _dense_prefix(self, index: int, L: _Layout, s_q: float, s_k: float) -> None:
        """Prefix queries x every key and the other queries x prefix keys, both dense on FA4 (output o1, LSE)."""
        from flash_attn.cute.interface import _flash_attn_fwd
        b, n_pre, rows = L.buffers, L.n_pre, L.rows
        q, kk = b["qc"][None], b["kc"][None]
        o1, sm = b["o1"], 1.0 / math.sqrt(self.head_dim)
        extra: dict[str, Any] = {}
        if self.dense_prefix == "fp8":
            ext().v_to_fp8(b["vc"], b["vs_head"], b["vc8"])
            v = b["vc8"][None]
            descale = self.__dict__.setdefault("_descale_cache", {}).get(index)
            if descale is None:
                full = lambda x: torch.full((1, self.heads), x, device=o1.device, dtype=torch.float32)  # noqa: E731
                descale = self._descale_cache[index] = (full(s_q), full(s_k))
            extra = dict(q_descale=descale[0], k_descale=descale[1], v_descale=b["vs_head"].view(1, self.heads))
        else:
            v = b["vc"][None]
        _flash_attn_fwd(q[:, :n_pre], kk, v, softmax_scale=sm, return_lse=True, out=o1[:, :n_pre], lse=b["lse1a"],
                        **extra)
        _flash_attn_fwd(q[:, n_pre:], kk[:, :n_pre], v[:, :n_pre], softmax_scale=sm, return_lse=True,
                        out=o1[:, n_pre:], lse=b["lse1b"], **extra)


    def _lora_add(self, layer: Any, a: torch.Tensor, out: torch.Tensor, sf: torch.Tensor | None = None) -> None:
        """Add ``layer``'s side-branch LoRA adapter to its GEMM output. ``a`` is the GEMM's input (FP8, or NVFP4 with
        its block scales ``sf``) in the same row-scaled units as ``out`` (each row's scale is applied downstream), so
        the adapter sees what the base weight sees: out += scale * (a A^T) B^T."""
        adapter = _adapter(layer)
        if adapter is None:
            return
        A, B, scale = adapter
        cached = self._lora_dev.get(id(layer))
        if cached is None or cached[0] is not A:
            # The pipeline may hold the factors on the host; copy them once per adapter, not per forward.
            cached = (A, A.to(out.device, torch.bfloat16).t().contiguous(), B.to(out.device, torch.bfloat16).t().contiguous())
            self._lora_dev[id(layer)] = cached
        # q, k and v read the same input and (community LoRAs split one fused qkv adapter) share A: x A^T once.
        key = (a.data_ptr(), A.data_ptr(), tuple(A.shape))
        if self._lora_t is not None and self._lora_t[0] == key:
            t = self._lora_t[1]
        else:
            if sf is None:
                x = a.to(torch.bfloat16)
            else:
                x = torch.empty(a.shape[0], a.shape[1] * 2, device=a.device, dtype=torch.bfloat16)
                ext().fp4_dequant(a, sf, x)
            t = x @ cached[1]
            self._lora_t = (key, t)
        out.addmm_(t, cached[2], alpha=scale)

    @torch.no_grad()
    def _merged_fp8(self, wrapper: Any, adapter: tuple, source: torch.Tensor | None,
                    quantize: Any) -> tuple[torch.Tensor, ...] | None:
        """FP8 weight and scale of ``source`` (the dense weight under ``wrapper``) with ``wrapper``'s active LoRA
        ``adapter`` merged in (in fp32) and quantized by ``quantize`` the way the base weight was, built on the first
        forward after the adapter or its strength changes; None without a dense source (the adapter then runs as a
        side branch)."""
        if source is None or not self.lora_merge:
            return None
        A, B, scale = adapter
        # Keyed by the adapter file and strength: FastVideo wraps the same factors in new Parameters on every strength
        # change. An adapter without a path is keyed by those Parameters, which the entry holds (ids stay unique).
        path = getattr(wrapper, "lora_path", None)
        key = (path, scale) if path else (wrapper.lora_A, wrapper.lora_B, scale)
        entry = self._merged.get(id(wrapper))
        if entry is None or len(entry[0]) != len(key) or any(
                old is not new and (torch.is_tensor(old) or old != new) for old, new in zip(entry[0], key)):
            device = self.one.device
            merged = source.to(device, non_blocking=True).float()
            merged.addmm_(B.to(device, torch.float32), A.to(device, torch.float32), alpha=scale)
            entry = self._merged[id(wrapper)] = (key, *quantize(merged))
            del merged
        return entry[1], entry[2]

    def _gemm(self, a: torch.Tensor, sf: torch.Tensor | None, layer: Any, out: torch.Tensor) -> None:
        wrapper, layer = layer, _base(layer)
        if sf is None:
            weight, scale = layer._fp8_weight, layer._fp8_weight_scale
            adapter = _adapter(wrapper)
            merged = None
            if adapter is not None:
                merged = self._merged_fp8(wrapper, adapter, getattr(layer, "_fp8_source", None),
                                          lambda w: _on_grid(w, layer._fp8_weight_scale))
                if merged is not None:
                    weight, scale = merged
            torch._scaled_mm(a, weight.t(), scale_a=self.one, scale_b=scale, out_dtype=torch.bfloat16, out=out)
            if adapter is not None and merged is None:
                self._lora_add(wrapper, a, out)
            return
        alpha = getattr(layer, "_h3fast_alpha", None)
        if alpha is None:
            alpha = layer._nvfp4_alpha if hasattr(layer, "_nvfp4_alpha") else 1.0 / layer._weight_global_sf
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, device=out.device, dtype=torch.float32)
            layer._h3fast_alpha = alpha = alpha.float().reshape(1)
        weight = layer._nvfp4_weight
        weight = weight.view(torch.uint8) if weight.dtype == torch.float4_e2m1fn_x2 else weight
        # FlashInfer directly: the registered custom op refuses an out= buffer (aliasing check).
        _require_flashinfer()[1](a, weight.T, sf, layer._nvfp4_weight_scale.T, alpha, torch.bfloat16, out,
                                 block_size=16, use_8x4_sf_layout=False, backend=_mm_fp4_backend("h3_dit"))
        self._lora_add(wrapper, a, out, sf)

    def forward(self, packed: torch.Tensor, temb: torch.Tensor, adaln_indices: torch.Tensor,
                rotary: tuple[torch.Tensor, torch.Tensor], meta: Any, reduction: tuple | None = None) -> torch.Tensor:
        """All blocks. ``reduction`` = (token-reduction plan, its rotary, its AdaLN indices): blocks [begin, end) run on
        horizontally paired video rows in a second layout (see fastvideo/models/dits/minimax_h3_token_reduction.py)."""
        L = self._get_layout(packed.shape[1], packed.device, meta, rotary, adaln_indices)
        cfg = {id(L): self._attn_cfg(meta)}
        R = maps = plan = None
        begin = end = len(self.blocks)
        if reduction is not None:
            plan, r_rotary, r_adaln = reduction
            R = self._get_layout(plan.reduced_length, packed.device, plan.attn_metadata, r_rotary, r_adaln, False)
            cfg[id(R)] = self._attn_cfg(plan.attn_metadata)
            maps = self._pair_maps(plan, L, R)
            begin, end = plan.begin, plan.end
        torch.index_select(packed[0], 0, L.perm, out=L.buffers["x"])
        cur, tabs, norm_in = L, self.blocks[0].adaln_proj(temb), True
        for index, block in enumerate(self.blocks):
            if R is not None and index == begin:
                self._pool(L, R, maps)
                cur, norm_in = R, True
            if R is not None and index == end:
                self._expand(L, R, maps, plan.update_scale)
                cur, norm_in = L, True
            if norm_in:
                b = cur.buffers
                ext().resid_norm_quant(b["x"], None, None, None, cur.adaln, block.norm1.weight, tabs[0].to(b["x"].dtype),
                                       tabs[1].to(b["x"].dtype), b["x"], b["h"], b.get("h_sf"), b["hs"], block.norm1.eps)
            switch = R is not None and index + 1 in (begin, end)
            tabs, norm_in = self._block(index, block, cur, cfg[id(cur)], tabs, temb, switch)
        if R is not None and cur is R:
            self._expand(L, R, maps, plan.update_scale)
        return L.buffers["x"].index_select(0, L.inv).unsqueeze(0)

    @staticmethod
    def _attn_cfg(meta: Any) -> dict[str, Any]:
        spans = meta.video_tile_spans
        return {"starts": [s for s, _ in spans], "ends": [e for _, e in spans],
                "sparse_ks": [compute_topk(sp, e - s) for (s, e), sp in zip(spans, meta.span_sparsities, strict=True)],
                "dense_ks": [e - s for s, e in spans], "prefix_tiles": meta.num_prefix_tiles,
                "dense_layers": meta.dense_layers}

    def _block(self, index: int, block: Any, L: _Layout, cfg: dict[str, Any], tabs: Any, temb: torch.Tensor,
               switch: bool) -> tuple[Any, bool]:
        """One block on layout L; returns the next block's AdaLN tables and whether the next block still needs its
        input norm (the last block, or the next one runs on the other token-reduction layout)."""
        k = ext()
        b = L.buffers
        x, y, q, kk, v, g, u = b["x"], b["y"], b["q"], b["k"], b["v"], b["g"], b["u"]
        h, hs, a, as_, f, fs = b["h"], b["hs"], b["a"], b["as"], b["f"], b["fs"]
        h_sf, a_sf, f_sf = b.get("h_sf"), b.get("a_sf"), b.get("f_sf")
        fwd = _vsa_forward(L.tile)
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (t.to(x.dtype) for t in tabs)
        attn = block.attn
        self._lora_t = None
        self._gemm(h, h_sf, attn.to_q, q)
        self._gemm(h, h_sf, attn.to_k, kk)
        self._gemm(h, h_sf, attn.to_v, v)
        gate_entry = self._layer_gate(index, attn)
        main = torch.cuda.current_stream()
        side = self.side if gate_entry is not None else None
        if side is not None:
            side.wait_stream(main)
        if gate_entry is not None:
            adapter = merged = None
            if not self.fp4:  # the merged weight is built on the main stream (see _merged_fp8)
                adapter = _adapter(attn.to_gate_compress)
                merged = None if adapter is None else self._merged_fp8(
                    attn.to_gate_compress, adapter, _base(attn.to_gate_compress).weight,
                    lambda w: _on_grid(w, gate_entry[1]))
            with torch.cuda.stream(side or main):
                self._lora_t = None
                if self.fp4:
                    self._gemm(h, h_sf, gate_entry, g)
                else:
                    weight, scale = merged or gate_entry
                    torch._scaled_mm(h, weight.t(), scale_a=self.one, scale_b=scale, out_dtype=torch.bfloat16, out=g)
                    if adapter is not None and merged is None:
                        self._lora_add(attn.to_gate_compress, h, g)
        self._lora_t = None  # the next side branches read other buffers (and h gets new contents before fc_in)
        split = L.n_pre > 0
        if self.fp8_attn:
            s_q, s_k = self._qk_scales(index, attn)
            b["vs_head"].zero_()
            k.qkv_prologue8(q, kk, v, hs, L.tile_row0, L.vbs, attn.norm_q.weight, attn.norm_k.weight, L.cos, L.sin,
                            1.0 / s_q, 1.0 / s_k, b["qb"], b["kb"], b["vb"], b["vs_blk"], b["vs_head"], b["qp"],
                            b["kp"], b["vp"], self.heads, L.n_tiles, L.tile, attn.norm_q.eps, b.get("qc"),
                            b.get("kc"), b.get("vc"))
        else:
            k.qkv_prologue(q, kk, v, hs, L.tile_row0, L.vbs, attn.norm_q.weight, attn.norm_k.weight, L.cos, L.sin,
                           b["qb"], b["kb"], b["vb"], b["qp"], b["kp"], b["vp"], self.heads, L.n_tiles, L.tile,
                           attn.norm_q.eps, b.get("qc"), b.get("kc"), b.get("vc"))
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32_scores
        torch.baddbmm(b["sc"], b["qp"], b["kp"].transpose(1, 2), beta=0.0, alpha=sm_scale, out=b["sc"])
        dense = index in cfg["dense_layers"]
        k.topk_rows(b["sc"], L.n_tiles, L.n_ext, cfg["prefix_tiles"], cfg["starts"], cfg["ends"],
                    cfg["dense_ks"] if dense else cfg["sparse_ks"], b["idx"], b["num"], split)
        oc, lse2 = None, b.get("lse2")
        if gate_entry is not None:
            if side is not None:
                side.wait_stream(main)
            with torch.cuda.stream(side or main):
                oc = torch.bmm(torch.softmax(b["sc"], dim=-1), b["vp"]).to(x.dtype)
            if side is not None:
                oc.record_stream(main)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        if L.tile == 128 and self.pair128:
            # each 128-key block as two 64-key blocks selected by both halves of the 128-row tile
            torch.stack([2 * b["idx"], 2 * b["idx"] + 1], -1, out=b["uidx"].view(*b["idx"].shape, 2))
            torch.mul(b["num"], 2, out=b["unum"])
            eight = self.fp8_attn
            scale = sm_scale * (s_q * s_k if eight and not self.qk16 else 1.0)
            if split:
                self._dense_prefix(index, L, s_q, s_k)
            out = k.vsa_pair(b["qb"], b["kb"], b["vb"], b["uidx"], b["umem"], b["unum"], b["vbs64"], scale,
                             b["vs_blk"] if eight else None, b["vs_head"] if eight else None, b.get("lse2"))
        elif self.fp8_attn:
            coef = sm_scale * math.log2(math.e) * (1.0 if self.qk16 else s_q * s_k)
            if split:
                self._dense_prefix(index, L, s_q, s_k)
            out = k.vsa_fp8(b["qb"], b["kb"], b["vb"], b["idx"], b["num"], L.vbs_ext, coef, b["vs_blk"],
                            b["vs_head"], b.get("lse2"))
        elif split:
            self._dense_prefix(index, L, 1.0, 1.0)
            out, lse2 = fwd(b["qb"], b["kb"], b["vb"], None, b["idx"], b["num"], L.vbs_ext, sm_scale, True)
        else:
            out = fwd(b["qb"], b["kb"], b["vb"], None, b["idx"], b["num"], L.vbs_ext, sm_scale, False)[0]
        if side is not None:
            main.wait_stream(side)
        k.attn_gate_quant(out, g if gate_entry is not None else None, hs if gate_entry is not None else None, oc, L.slot,
                          L.n_tiles, L.tile, a, a_sf, as_, L.rows, b.get("o1"), b.get("lse1a"), b.get("lse1b"),
                          lse2)
        self._gemm(a, a_sf, attn.to_out, y)
        k.resid_norm_quant(x, y, as_, gate_msa, L.adaln, block.norm2.weight, shift_mlp, scale_mlp, x, h, h_sf, hs,
                           block.norm2.eps)
        self._gemm(h, h_sf, block.ff.fc_in, u)
        k.swiglu_quant(u, hs, f, f_sf, fs)
        self._gemm(f, f_sf, block.ff.fc_out, y)
        if index + 1 < len(self.blocks):
            nxt = self.blocks[index + 1]
            tabs = nxt.adaln_proj(temb)
            if not switch:
                k.resid_norm_quant(x, y, fs, gate_mlp, L.adaln, nxt.norm1.weight, tabs[0].to(x.dtype),
                                   tabs[1].to(x.dtype), x, h, h_sf, hs, nxt.norm1.eps)
                return tabs, False
        k.resid_norm_quant(x, y, fs, gate_mlp, L.adaln, None, None, None, x, None, None, None, 0.0)
        return tabs, True

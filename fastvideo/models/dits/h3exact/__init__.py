"""Bit-exact fused q/k RMSNorm + RoPE prefix for MiniMax-H3's standard (non-fused) path, i.e. the full 50-step model.

Today's path runs nn.RMSNorm on q and on k, then FastVideo's exact RoPE prefix on each: four passes over q and k.
qknorm_rope.cu does all of it in one in-place pass and reproduces the same arithmetic bit for bit (tests/
test_qknorm_rope.py), 3.47 -> 0.90 ms per layer at 768p. Built on first use into $FASTVIDEO_H3_FAST_BUILD/h3exact;
FASTVIDEO_H3_EXACT_QK=0 turns it off. If the build fails the caller keeps today's path."""
import os
from pathlib import Path

import torch

_ext = None
_tried = False


def _op():
    global _ext, _tried
    if _tried:
        return _ext
    _tried = True
    try:
        from torch.utils.cpp_extension import load
        build = Path(os.environ.get("FASTVIDEO_H3_FAST_BUILD", "/cache/h3fast")) / "h3exact"
        build.mkdir(parents=True, exist_ok=True)
        _ext = load(name="h3exact_qknorm_rope", sources=[str(Path(__file__).with_name("qknorm_rope.cu"))],
                    build_directory=str(build), verbose=False,
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"])
    except Exception as e:  # noqa: BLE001 - any build problem falls back to the unfused path
        print(f" -- h3exact: fused q/k norm + RoPE unavailable ({type(e).__name__}: {e})", flush=True)
        _ext = None
    return _ext


def try_qknorm_rope(query, key, norm_q, norm_k, rotary_emb) -> bool:
    """Normalize and rotate query and key in place. True when it ran; False leaves them untouched for today's path."""
    if os.environ.get("FASTVIDEO_H3_EXACT_QK", "1") != "1" or rotary_emb is None:
        return False
    cos, sin = rotary_emb
    if not (query.is_cuda and query.dtype == key.dtype == torch.bfloat16 and query.dim() == 4
            and query.shape == key.shape and query.shape[0] == 1 and query.shape[-1] == 128 and cos.shape[-1] == 96
            and query.stride(-1) == 1 and key.stride(-1) == 1
            and all(s % 4 == 0 for s in (query.stride(1), query.stride(2), key.stride(1), key.stride(2)))
            and query.data_ptr() % 8 == 0 and key.data_ptr() % 8 == 0
            and norm_q.weight is not None and norm_k.weight is not None and norm_q.eps == norm_k.eps
            and norm_q.weight.dtype == norm_k.weight.dtype == torch.bfloat16
            and not torch.compiler.is_compiling()):
        return False
    ext = _op()
    if ext is None:
        return False
    ext.qknorm_rope(query, key, norm_q.weight, norm_k.weight, cos.to(torch.bfloat16).contiguous(),
                    sin.to(torch.bfloat16).contiguous(), norm_q.eps)
    return True

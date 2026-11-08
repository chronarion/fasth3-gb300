"""Unit checks of the fused h3fast kernels against PyTorch formulas (random data at the real FastH3 widths).

Run inside the image, on the GB300:
    docker run --rm --device "$FASTH3_GPU" --entrypoint /opt/venv/bin/python fasth3-gb300 /tests/test_kernels.py
(mount this directory at /tests). Prints the relative error and time of every kernel; the errors should be at the level
of the output format (FP8 ~2-3%, NVFP4 ~8-10%, BF16 < 0.5%).
"""
import math, torch
from fastvideo.models.dits.h3fast import ext
k = ext()
torch.manual_seed(0); dev = "cuda"
C, H, D, F = 5376, 56, 128, 14336
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
def dq(q, s): return q.float() * s[:, None]
def bench(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
# geometry: variable tiles
tile = 64; n = 668
vbs = torch.full((n,), 64, device=dev, dtype=torch.int32); vbs[::10] = 16; vbs[3::11] = 32; vbs[0] = 16
S = int(vbs.sum())
tile_row0 = (vbs.cumsum(0) - vbs).to(torch.int32)
slot = torch.cat([torch.arange(int(v), device=dev) + i * tile for i, v in enumerate(vbs.tolist())]).to(torch.int32)
idx = torch.randint(0, 6, (S,), device=dev, dtype=torch.int32)
x = torch.randn(S, C, device=dev, dtype=torch.bfloat16) * 3
y = torch.randn(S, C, device=dev, dtype=torch.bfloat16)
sy = torch.rand(S, device=dev) + 0.5
tabs = torch.randn(6, 6 * C, device=dev, dtype=torch.bfloat16) * 0.3
sh, sc, gt = tabs[:, :C], tabs[:, C:2 * C], tabs[:, 2 * C:3 * C]
w = torch.randn(C, device=dev, dtype=torch.bfloat16)
xo = torch.empty_like(x); h = torch.empty(S, C, device=dev, dtype=torch.float8_e4m3fn); hs = torch.empty(S, device=dev)
k.resid_norm_quant(x, y, sy, gt, idx, w, sh, sc, xo, h, None, hs, 1e-5)
il = idx.long()
ref_x = (x.float() + gt[il].float() * (sy[:, None] * y.float())).bfloat16()
m = ref_x.float() * torch.rsqrt(ref_x.float().pow(2).mean(-1, keepdim=True) + 1e-5) * w.float() * (1 + sc[il].float()) + sh[il].float()
print("resid_norm_quant(2-row) x", rel(xo, ref_x), "h", rel(dq(h, hs), m), "ms", bench(lambda: k.resid_norm_quant(x, y, sy, gt, idx, w, sh, sc, xo, h, None, hs, 1e-5)))
# qkv prologue
W = H * D
q = torch.randn(S, W, device=dev, dtype=torch.bfloat16); kk = torch.randn(S, W, device=dev, dtype=torch.bfloat16); v = torch.randn(S, W, device=dev, dtype=torch.bfloat16)
s1 = torch.rand(S, device=dev) + 0.5
wq = torch.randn(D, device=dev, dtype=torch.bfloat16); wk = torch.randn(D, device=dev, dtype=torch.bfloat16)
ang = torch.rand(S, 48, device=dev) * 6.28; cosb = torch.cat([ang.cos()] * 2, -1).bfloat16(); sinb = torch.cat([ang.sin()] * 2, -1).bfloat16()
qb = torch.zeros(1, H, n * tile, D, device=dev, dtype=torch.bfloat16); kb = torch.zeros_like(qb); vb = torch.zeros_like(qb)
qp = torch.empty(H, n, D, device=dev); kp = torch.empty_like(qp); vp = torch.empty_like(qp)
args = (q, kk, v, s1, tile_row0, vbs, wq, wk, cosb, sinb, qb, kb, vb, qp, kp, vp, H, n, tile, 1e-5)
k.qkv_prologue(*args)
def rope(t):
    t = t.float().view(S, H, D); r, p = t[..., :96], t[..., 96:]
    a, b = r[..., :48], r[..., 48:]
    return torch.cat([r * cosb.float()[:, None] + torch.cat([-b, a], -1) * sinb.float()[:, None], p], -1)
def nrm(t, wt):
    t = t.float().view(S, H, D) * s1[:, None, None]
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5) * wt.float()
rq, rk = rope(nrm(q, wq)), rope(nrm(kk, wk)); rv = v.float().view(S, H, D) * s1[:, None, None]
got = lambda b: b[0].index_select(1, slot.long()).permute(1, 0, 2)
print("qkv q", rel(got(qb), rq), "k", rel(got(kb), rk), "v", rel(got(vb), rv), "ms", bench(lambda: k.qkv_prologue(*args)))
pad = torch.zeros(n * tile, H, D, device=dev); pad[slot.long()] = rq
pool = pad.view(n, tile, H, D).sum(1) / vbs.float()[:, None, None]
print("pool q", rel(qp, pool.permute(1, 0, 2)))
# topk
sco = torch.randn(H, n, n, device=dev)
ix = torch.zeros(1, H, n, n, device=dev, dtype=torch.int32); nm = torch.zeros(1, H, n, device=dev, dtype=torch.int32)
prefix = 8; kk_ = math.ceil(0.2 * (n - prefix))
k.topk_rows(sco, n, n, prefix, [prefix], [n], [kk_], ix, nm, False)
ref_i = sco[..., prefix:].topk(kk_, dim=-1).indices + prefix
mask = torch.zeros(H, n, n, device=dev, dtype=torch.bool).scatter_(-1, ref_i, True); mask[..., :prefix] = True; mask[:, :prefix, :] = True
cnt = nm[0]
ar = torch.arange(n, device=dev)
vals = ix[0].long().clamp(0, n - 1); keep = ar[None, None, :] < cnt[..., None]
g3 = torch.zeros(H, n, n, device=dev, dtype=torch.int32).scatter_add_(-1, vals, keep.int())
print("topk equal", bool(((g3 > 0) == mask).all()), "dup", int((g3 > 1).sum()), "counts", int(cnt[:, prefix:].min()), int(cnt[:, prefix:].max()), int(cnt[:, 0].min()),
      "sorted", bool(((ix[0][:, prefix:, 1:kk_ + prefix] - ix[0][:, prefix:, :kk_ + prefix - 1]) > 0).all()), "ms", bench(lambda: k.topk_rows(sco, n, n, prefix, [prefix], [n], [kk_], ix, nm, False)))
# attn gate quant
ob = torch.randn(1, H, n * tile, D, device=dev, dtype=torch.bfloat16); g = torch.randn(S, W, device=dev, dtype=torch.bfloat16); oc = torch.randn(H, n, D, device=dev, dtype=torch.bfloat16)
a = torch.empty(S, W, device=dev, dtype=torch.float8_e4m3fn); sa = torch.empty(S, device=dev)
k.attn_gate_quant(ob, g, s1, oc, slot, n, tile, a, None, sa, S, None, None, None, None)
oo = ob[0].index_select(1, slot.long()).permute(1, 0, 2).float()
ref = oo + (g.float().view(S, H, D) * s1[:, None, None]) * oc.permute(1, 0, 2).float().index_select(0, (slot // tile).long())
print("attn_gate", rel(dq(a, sa).view(S, H, D), ref), "ms", bench(lambda: k.attn_gate_quant(ob, g, s1, oc, slot, n, tile, a, None, sa, S, None, None, None, None)))
# swiglu
u = torch.randn(S, 2 * F, device=dev, dtype=torch.bfloat16); f = torch.empty(S, F, device=dev, dtype=torch.float8_e4m3fn); fs = torch.empty(S, device=dev)
k.swiglu_quant(u, s1, f, None, fs)
us = u.float() * s1[:, None]; ref = us[:, :F] * torch.nn.functional.silu(us[:, F:])
print("swiglu", rel(dq(f, fs), ref), "ms", bench(lambda: k.swiglu_quant(u, s1, f, None, fs)))
# NVFP4 outputs against flashinfer's quantizer semantics
from flashinfer import nvfp4_quantize, SfLayout
pad = (S + 127) // 128 * 128
def dq4(q, sf, row_s, Kc):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    lo, hi = lut[(q & 15).long()], lut[(q >> 4).long()]
    v = torch.stack([lo, hi], -1).flatten(1)                        # [S, K]
    r = torch.arange(S, device=dev)[:, None]; c = torch.arange(Kc // 16, device=dev)[None, :]
    off = (r // 128) * (Kc // 64) * 512 + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
    sc = sf.flatten()[off].view(torch.float8_e4m3fn).float()
    return v.view(S, Kc // 16, 16) * sc[..., None] * row_s[:, None, None]
h4 = torch.empty(S, C // 2, device=dev, dtype=torch.uint8); h4s = torch.zeros(pad, C // 16, device=dev, dtype=torch.uint8); hs4 = torch.empty(S, device=dev)
k.resid_norm_quant(x, y, sy, gt, idx, w, sh, sc, xo, h4, h4s, hs4, 1e-5)
print("fp4 resid_norm_quant", rel(dq4(h4, h4s, hs4, C).flatten(1), m), "ms", bench(lambda: k.resid_norm_quant(x, y, sy, gt, idx, w, sh, sc, xo, h4, h4s, hs4, 1e-5)))
gs = (448 * 6) / m.abs().amax()
ref_q, ref_s = nvfp4_quantize(m.bfloat16(), gs.float().view(1), sfLayout=SfLayout.layout_128x4, do_shuffle=False)
print("   flashinfer per-tensor error for comparison", rel(dq4(ref_q.view(torch.uint8), ref_s.view(torch.uint8), torch.full((S,), 1 / gs.item(), device=dev), C).flatten(1), m))
f4 = torch.empty(S, F // 2, device=dev, dtype=torch.uint8); f4s = torch.zeros(pad, F // 16, device=dev, dtype=torch.uint8); fs4 = torch.empty(S, device=dev)
k.swiglu_quant(u, s1, f4, f4s, fs4)
us = u.float() * s1[:, None]; ref = us[:, :F] * torch.nn.functional.silu(us[:, F:])
print("fp4 swiglu", rel(dq4(f4, f4s, fs4, F).flatten(1), ref), "ms", bench(lambda: k.swiglu_quant(u, s1, f4, f4s, fs4)))
a4 = torch.empty(S, W // 2, device=dev, dtype=torch.uint8); a4s = torch.zeros(pad, W // 16, device=dev, dtype=torch.uint8); sa4 = torch.empty(S, device=dev)
k.attn_gate_quant(ob, g, s1, oc, slot, n, tile, a4, a4s, sa4, S, None, None, None, None)
oo = ob[0].index_select(1, slot.long()).permute(1, 0, 2).float()
ref = oo + (g.float().view(S, H, D) * s1[:, None, None]) * oc.permute(1, 0, 2).float().index_select(0, (slot // tile).long())
print("fp4 attn_gate", rel(dq4(a4, a4s, sa4, W).flatten(1).view(S, H, D), ref), "ms", bench(lambda: k.attn_gate_quant(ob, g, s1, oc, slot, n, tile, a4, a4s, sa4, S, None, None, None, None)))

"""Bit-exactness of the fused q/k RMSNorm + RoPE kernel (fastvideo/models/dits/h3exact) against the path it replaces:
torch nn.RMSNorm(128) on q and k, then FastVideo's exact RoPE prefix (96 of 128 channels rotated).

Run inside the image, on the GB300:
    docker run --rm --device "$FASTH3_GPU" -v "$PWD/tests:/tests" --entrypoint /opt/venv/bin/python fasth3-gb300 \
        /tests/test_qknorm_rope.py
Every output must match bit for bit (contiguous and strided inputs, small, large and heavy-tailed values); exit code 1
otherwise. Also prints the time of both paths at the 768p sequence length.
"""
import os, sys, tempfile, time
import torch

os.environ.setdefault("FASTVIDEO_H3_FAST_BUILD", tempfile.mkdtemp())
from fastvideo.models.dits import h3exact
from fastvideo.models.dits.minimax_h3_fusions import exact as ex

ext = h3exact._op()
assert ext is not None, "the kernel did not build"
dev = torch.device("cuda")
torch.manual_seed(0)
bad = cases = 0
for S, H, scale, tail in [(512, 56, 1.0, False), (2048, 56, 0.05, False), (1024, 56, 8.0, False), (777, 56, 1.0, True),
                          (4096, 56, 3.0, True), (300, 56, 1e-3, False), (1500, 56, 40.0, True)]:
    for strided in (False, True):
        base = torch.randn(1, S, (3 if strided else 1) * H * 128, device=dev) * scale
        if tail:
            base = base * torch.exp(torch.randn_like(base) * 1.5)          # some channels huge, some tiny
        base = base.to(torch.bfloat16)
        q0 = base[..., : H * 128].unflatten(-1, (H, 128))                  # strided: token stride 3*H*128, as in q|k|v
        k0 = (base[..., H * 128: 2 * H * 128] if strided else (base * 0.7).to(torch.bfloat16)).unflatten(-1, (H, 128))
        nq = torch.nn.RMSNorm(128, eps=1e-5).to(dev, torch.bfloat16); nq.weight.data = (torch.rand(128, device=dev) * 2).to(torch.bfloat16)
        nk = torch.nn.RMSNorm(128, eps=1e-5).to(dev, torch.bfloat16); nk.weight.data = (torch.rand(128, device=dev) * 2).to(torch.bfloat16)
        ang = torch.rand(S, 96, device=dev) * 50
        cos, sin = ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)
        ref_q = ex.rope_prefix(nq(q0), (cos, sin))
        ref_k = ex.rope_prefix(nk(k0), (cos, sin))
        if strided:
            buf = base.clone()
            q1 = buf[..., : H * 128].unflatten(-1, (H, 128)); k1 = buf[..., H * 128: 2 * H * 128].unflatten(-1, (H, 128))
        else:
            q1, k1 = q0.clone(), k0.clone()
        assert h3exact.try_qknorm_rope(q1, k1, nq, nk, (cos, sin))
        torch.cuda.synchronize()
        for a, b in ((q1, ref_q), (k1, ref_k)):
            cases += 1
            same = torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
            if not same:
                bad += 1
                n = int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())
                print(f"  S={S} scale={scale} tail={tail} strided={strided}: {n} values differ", flush=True)

S, H = 37296, 56                                                        # 768p, 124 frames
x = torch.randn(1, S, H * 128, device=dev).to(torch.bfloat16).unflatten(-1, (H, 128))
y = x.clone()
ang = torch.rand(S, 96, device=dev) * 50
cos, sin = ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)
n = torch.nn.RMSNorm(128, eps=1e-5).to(dev, torch.bfloat16)


def ms(f, it=20):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it * 1e3


t0 = ms(lambda: (ex.rope_prefix(n(x), (cos, sin)), ex.rope_prefix(n(y), (cos, sin))))
t1 = ms(lambda: ext.qknorm_rope(x, y, n.weight, n.weight, cos, sin, 1e-5))
print(f"q+k RMSNorm + RoPE at {S} tokens: unfused {t0:.2f} ms, fused {t1:.2f} ms")
print(f"bit-exact: {cases - bad} of {cases} outputs", flush=True)
sys.exit(1 if bad else 0)

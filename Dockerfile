# FastH3 fused inference path for one GB300 (sm_103a), as a layer over FastVideo's own image.
#
# 1. Base: FastVideo at commit a01c50a, built with its Dockerfile for sm_103a:
#      git clone https://github.com/hao-ai-lab/FastVideo && cd FastVideo && git checkout a01c50a
#      docker build -f docker/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST=10.3a \
#        --build-arg CMAKE_BUILD_PARALLEL_LEVEL=32 -t fastvideo:gb300-a01c50a .
# 2. This layer, from the root of this repository:
#      docker build -t fasth3-gb300 .
#
# The CUDA sources in fastvideo/models/dits/h3fast JIT-build on first use into $FASTVIDEO_H3_FAST_BUILD (the compose
# file mounts a cache volume there, so the build survives restarts).
ARG BASE=fastvideo:gb300-a01c50a
FROM ${BASE}

# FlashInfer's NVFP4 GEMMs need nvidia-cutlass-dsl >= 4.6.2, while FastVideo pins FlashAttention-4 (CuTe) to a commit that
# needs 4.6.0.dev0: move both to cutlass-dsl 4.8.1 and the FA4 release built for it. ninja: FlashInfer JIT-builds its
# NVFP4 quantize kernels on first use.
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps \
      "nvidia-cutlass-dsl==4.8.1" "nvidia-cutlass-dsl-libs-base==4.8.1" "flash-attn-4==4.0.0b36" "ninja"

# The image's Python (uv-managed, under /root/.local) is world-readable, but /root itself is not: open it so the server
# can run as an unprivileged user.
RUN chmod 755 /root

# The fused path: a new package plus small edits to FastVideo (see patches/).
COPY fastvideo/ /FastVideo/fastvideo/
COPY patches/fastvideo-a01c50a.patch /tmp/fastvideo-a01c50a.patch
RUN cd /FastVideo && git apply /tmp/fastvideo-a01c50a.patch && rm /tmp/fastvideo-a01c50a.patch

# The venv's bin (ninja for the JIT builds) first on PATH.
ENV PATH=/opt/venv/bin:${PATH} TORCH_CUDA_ARCH_LIST=10.3a FASTVIDEO_H3_FAST_BUILD=/cache/h3fast

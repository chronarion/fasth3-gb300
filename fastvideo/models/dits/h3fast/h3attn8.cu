// SPDX-License-Identifier: Apache-2.0
// Torch binding of the FP8 VSA forward (vsa_fp8.cuh).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "vsa_fp8.cuh"
#include "vsa_pair.cuh"

torch::Tensor vsa_fp8_op(torch::Tensor q8, torch::Tensor k8, torch::Tensor v8, torch::Tensor q2k_idx,
                         torch::Tensor q2k_num, torch::Tensor vbs, double qk_coef, torch::Tensor v_scale_blk,
                         torch::Tensor v_scale_head) {
  const at::cuda::OptionalCUDAGuard guard(device_of(q8));
  TORCH_CHECK(q8.dim() == 4 && q8.size(-1) == vsa_fp8::HEAD_DIM && q8.is_contiguous() && k8.is_contiguous());
  TORCH_CHECK(k8.sizes() == q8.sizes() && v8.sizes() == q8.sizes() && v8.scalar_type() == at::kByte);
  const bool qk16 = q8.scalar_type() == at::kBFloat16;
  TORCH_CHECK(qk16 || q8.scalar_type() == at::kByte);
  const int64_t B = q8.size(0), H = q8.size(1), S = q8.size(2);
  auto out = torch::empty({B, H, S, vsa_fp8::HEAD_DIM}, q8.options().dtype(torch::kBFloat16));
  vsa_fp8::Args a{};
  a.qk16 = qk16;
  a.q = q8.data_ptr();
  a.k = k8.data_ptr();
  a.v = v8.data_ptr<uint8_t>();
  a.o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.q2k_idx = q2k_idx.data_ptr<int>();
  a.q2k_num = q2k_num.data_ptr<int>();
  a.variable_block_sizes = vbs.data_ptr<int>();
  a.qk_coef = (float)qk_coef;
  a.v_scale_blk = v_scale_blk.data_ptr<float>();
  a.v_scale_head = v_scale_head.data_ptr<float>();
  a.batch = (int)B;
  a.num_heads = (int)H;
  a.seqlen = (int)S;
  a.num_blocks = (int)vbs.numel();
  a.max_kv = (int)q2k_idx.size(-1);
  const cudaError_t e = vsa_fp8::launch(a, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "vsa_fp8 launch failed: ", cudaGetErrorString(e));
  return out;
}

torch::Tensor vsa_pair_op(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor u_idx, torch::Tensor u_mem,
                          torch::Tensor u_num, torch::Tensor vbs, double sm_scale, std::optional<torch::Tensor> v_scale_blk,
                          std::optional<torch::Tensor> v_scale_head, std::optional<torch::Tensor> lse) {
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  TORCH_CHECK(q.dim() == 4 && q.size(-1) == vsa_pair::HEAD_DIM && q.is_contiguous());
  const bool qk8 = q.scalar_type() == at::kByte;
  TORCH_CHECK(qk8 || q.scalar_type() == at::kBFloat16);
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() && k.is_contiguous() && v.is_contiguous());
  const bool pv8 = v.scalar_type() == at::kByte;
  TORCH_CHECK(!pv8 || (v_scale_blk.has_value() && v_scale_head.has_value()));
  auto out = torch::empty(q.sizes(), q.options().dtype(torch::kBFloat16));
  vsa_pair::Args a{};
  a.qk8 = qk8;
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  a.v = pv8 ? nullptr : reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  a.v8 = pv8 ? v.data_ptr<uint8_t>() : nullptr;
  a.v_scale_blk = pv8 ? v_scale_blk->data_ptr<float>() : nullptr;
  a.v_scale_head = pv8 ? v_scale_head->data_ptr<float>() : nullptr;
  a.o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.u_idx = u_idx.data_ptr<int>();
  a.u_mem = u_mem.data_ptr<uint8_t>();
  a.u_num = u_num.data_ptr<int>();
  a.variable_block_sizes = vbs.data_ptr<int>();
  a.batch = (int)q.size(0);
  a.num_heads = (int)q.size(1);
  a.seqlen = (int)q.size(2);
  a.max_u = (int)u_idx.size(-1);
  a.sm_scale = (float)sm_scale;
  a.lse = lse.has_value() ? lse->data_ptr<float>() : nullptr;
  const cudaError_t e = vsa_pair::launch(a, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "vsa_pair launch failed: ", cudaGetErrorString(e));
  return out;
}

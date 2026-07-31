#!/usr/bin/env python3
"""Standalone single-GPU repro to step into the DSA indexer forward
(megatron/bridge/models/glm5/tilelang/tilelang_indexer_fwd.py:149).

Run under pdb on node-1 or node-2 (any one GPU):
    source /workspace/users/zhe-li/repos/glm-sft/env.sh
    CUDA_VISIBLE_DEVICES=0 python -m pdb tools/debug_indexer.py
Then:
    b megatron/bridge/models/glm5/tilelang/tilelang_indexer_fwd.py:149
    c                       # runs to the allocation line
    p seq_len ; p seq_len_kv ; p heads ; p index_dim
    p [seq_len, seq_len_kv]                 # -> the logits shape
    p seq_len*seq_len_kv*4/2**30            # -> GiB of the fp32 buffer
    n / s                   # step over / into
Tweak SEQ_LEN / SEQ_LEN_KV below to see how the [S x S_kv] buffer scales.
"""
import torch
from megatron.bridge.models.glm5.tilelang.tilelang_indexer_fwd import indexer_fwd_interface

# GLM-5.2 indexer dims (from the HF config): index_n_heads=32, index_head_dim=128.
HEADS, INDEX_DIM = 32, 128
SEQ_LEN, SEQ_LEN_KV = 4096, 8192          # small so the kernel actually runs on 1 GPU

dev = "cuda"
q  = torch.randn(SEQ_LEN, HEADS, INDEX_DIM, device=dev, dtype=torch.bfloat16)
kv = torch.randn(SEQ_LEN_KV, INDEX_DIM,     device=dev, dtype=torch.bfloat16)
w  = torch.randn(SEQ_LEN, HEADS,            device=dev, dtype=torch.float32)
# per-query key window [start, end): causal single-doc example
ks = torch.zeros(SEQ_LEN, device=dev, dtype=torch.int32)
ke = torch.arange(1, SEQ_LEN + 1, device=dev, dtype=torch.int32).clamp(max=SEQ_LEN_KV)

print(f"calling indexer_fwd_interface: q={tuple(q.shape)} kv={tuple(kv.shape)} "
      f"-> logits will be [{SEQ_LEN}, {SEQ_LEN_KV}] fp32 "
      f"= {SEQ_LEN*SEQ_LEN_KV*4/2**30:.3f} GiB")

logits = indexer_fwd_interface(q, kv, w, ks, ke, clean_logits=True)   # <-- line 149 lives inside here
print("logits:", tuple(logits.shape), logits.dtype, "max", logits.max().item())

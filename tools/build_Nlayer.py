#!/usr/bin/env python
"""Build an N-layer decapitated copy of a GLM-5.2 checkpoint (for memory-scaling).

Mirrors yoshinari's build_10layer_bf16.py, parameterized by N and source:
  - keep tensors for layers 0..N-1 + all non-layer tensors (embed, norm, lm_head)
  - preserve original shard filenames; full shards hard-copied, leaky shards re-written
  - truncate config.json num_hidden_layers + per-layer list fields (incl DSA indexer_types)
  - copy tokenizer / chat_template / generation_config from the source
Preserves exact dtypes (works for FP8 or bf16 sources; no cast).

Usage: build_Nlayer.py --n 30 --src <hf_ckpt> --dst <out_dir>
"""
import argparse, json, os, re, shutil
from collections import Counter
import torch
from safetensors import safe_open
from safetensors.torch import save_file

DT_BYTES = {"BF16":2,"F16":2,"F32":4,"F64":8,"F8_E4M3":1,"F8_E5M2":1,
            "I8":1,"U8":1,"I16":2,"I32":4,"I64":8,"BOOL":1}
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    N, SRC, DST = a.n, a.src, a.dst
    os.makedirs(DST, exist_ok=True)

    def keep(name):
        m = LAYER_RE.search(name)
        return int(m.group(1)) < N if m else True

    with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
        src_wm = json.load(f)["weight_map"]
    keep_names = sorted(k for k in src_wm if keep(k))
    print(f"[select] keeping {len(keep_names)}/{len(src_wm)} tensors (N={N})", flush=True)

    by_shard = {}
    for name in keep_names:
        by_shard.setdefault(src_wm[name], []).append(name)
    full_count = Counter(src_wm.values())
    print(f"[select] shards involved: {len(by_shard)}", flush=True)

    new_wm, total_size = {}, 0
    def ecount(shape):
        n = 1
        for d in shape: n *= d
        return n
    for i, (shard, names) in enumerate(sorted(by_shard.items())):
        dst_path = os.path.join(DST, shard)
        is_full = len(names) == full_count[shard]
        with safe_open(os.path.join(SRC, shard), framework="pt", device="cpu") as fh:
            if is_full:
                for name in names:
                    sl = fh.get_slice(name)
                    total_size += ecount(sl.get_shape()) * DT_BYTES[sl.get_dtype()]
            else:
                buf = {}
                for name in names:
                    t = fh.get_tensor(name)
                    total_size += t.numel() * t.element_size()
                    buf[name] = t
        if is_full:
            shutil.copy2(os.path.join(SRC, shard), dst_path)
        else:
            save_file(buf, dst_path, metadata={"format": "pt"})
        for name in names:
            new_wm[name] = shard
        if i % 20 == 0:
            print(f"[write] {i+1}/{len(by_shard)} shards ({'full' if is_full else 'partial'})", flush=True)

    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_wm}, f, indent=2)
    print(f"[index] total_size={total_size/1e9:.1f} GB, {len(new_wm)} tensors", flush=True)

    with open(os.path.join(SRC, "config.json")) as f:
        cfg = json.load(f)
    cfg["num_hidden_layers"] = N
    for k in ["layer_types", "mlp_layer_types", "indexer_types"]:
        if isinstance(cfg.get(k), list):
            cfg[k] = cfg[k][:N]
    with open(os.path.join(DST, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"[config] num_hidden_layers -> {N}", flush=True)

    for fn in os.listdir(SRC):
        if fn in ("config.json", "model.safetensors.index.json") or fn.endswith(".safetensors"):
            continue
        sf = os.path.join(SRC, fn)
        if os.path.isfile(sf):
            shutil.copy2(sf, os.path.join(DST, fn))
    print(f"[done] -> {DST}", flush=True)

if __name__ == "__main__":
    main()

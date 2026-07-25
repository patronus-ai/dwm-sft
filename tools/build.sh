#!/usr/bin/env bash
# Self-contained, PUBLIC build of the Miles GLM-5.2 LoRA-SFT stack on B200 / cu13.
# Authored fresh for zhe-li (not copied from another user's tree). Everything
# under /workspace/users/zhe-li; caches are the cluster-shared /workspace/.cache.
#
# Sources (all PUBLIC, no private auth):
#   miles     : radixark/miles           (already cloned at $ROOT/miles)
#   Megatron  : radixark/Megatron-LM      @ miles-main (+ DSA-dispatch patch)
#   Bridge    : radixark/Megatron-Bridge  @ bridge
#   mbridge   : ISEEKYAN/mbridge          @ 89eb108
#   sglang    : sgl-project/sglang        @ sglang-miles (import-time dep of miles)
#   kernels   : yueming-yuan/miles-wheels @ cu130-x86_64-v0.5.12 (prebuilt, no compile)
#   tilelang  : pip (abi3 wheel)          -- REQUIRED: glm5.py hard-imports it
#
# Stamp-based + resumable: each stage runs once (touch $STAMP/<name>).
set -euo pipefail

ROOT=/workspace/users/zhe-li/repos/glm-sft
VENV=$ROOT/.venv
WHEELS=$ROOT/wheels
LOG=$ROOT/build-logs
STAMP=$ROOT/.build-stamps
WHEELS_TAG=cu130-x86_64-v0.5.12
WHEELS_REPO=yueming-yuan/miles-wheels
mkdir -p "$WHEELS" "$LOG" "$STAMP"

# zhe-li's OWN profile (routes caches to shared /workspace/.cache).
source /workspace/users/zhe-li/env/activate.sh
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.cache/uv}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.cache/pip}
PIP="uv pip install"

done_stage() { touch "$STAMP/$1"; }
need()       { [ ! -f "$STAMP/$1" ]; }

verify() { case "$1" in
    *.whl)    python3 -c "import zipfile;zipfile.ZipFile('$1').testzip()" 2>/dev/null ;;
    *.tar.gz) gzip -t "$1" 2>/dev/null ;; *) [ -s "$1" ] ;; esac ; }
dl() { local url=$1 dest=$2 i; for i in 1 2 3 4 5 6 7 8; do
    curl -fSL --retry 5 --retry-delay 3 --retry-all-errors -C - -o "$dest" "$url" || { sleep 3; continue; }
    verify "$dest" && return 0; rm -f "$dest"; sleep 2; done; echo "FAILED $url" >&2; return 1; }
git config --global http.postBuffer 1048576000 || true
gitclone() { local url=$1 dir=$2; shift 2; [ -d "$dir/.git" ] && return 0
  for i in 1 2 3 4 5 6 7 8; do git clone "$@" "$url" "$dir" && return 0; rm -rf "$dir"; sleep 5; done; return 1; }

# --- Stage 0: venv + torch ---------------------------------------------------
if need 00-venv; then
  echo "[stage0] venv + torch 2.11.0+cu130"
  uv venv --python 3.12 "$VENV"
  source "$VENV/bin/activate"
  $PIP --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 torchvision==0.26.0+cu130
  done_stage 00-venv
fi
source "$VENV/bin/activate"

# --- Stage 1: download prebuilt kernel wheels --------------------------------
if need 01-wheels-dl; then
  echo "[stage1] fetch wheels $WHEELS_TAG"
  curl -sL "https://api.github.com/repos/${WHEELS_REPO}/releases/tags/${WHEELS_TAG}" \
    | python3 -c "import sys,json;[print(a['name'],a['browser_download_url']) for a in json.load(sys.stdin).get('assets',[]) if a['name'].endswith(('.whl','.tar.gz'))]" \
    > "$WHEELS/manifest.txt"
  while read -r name url; do dl "$url" "$WHEELS/$name"; done < "$WHEELS/manifest.txt"
  done_stage 01-wheels-dl
fi
if need 02-wheels-install; then
  echo "[stage2] install prebuilt kernels"
  $PIP "$WHEELS"/flash_attn-*.whl
  $PIP "$WHEELS"/flash_attn_3-*.whl
  $PIP "$WHEELS"/apex-*.whl
  $PIP --no-deps "$WHEELS"/transformer_engine-*.whl
  $PIP transformer_engine_cu13==2.12.0
  $PIP "$WHEELS"/transformer_engine_torch-*.whl
  $PIP --force-reinstall "$WHEELS"/sglang_router-*.whl
  tar xzf "$WHEELS"/sgl-model-gateway-linux-*.tar.gz -C "$VENV/bin/" && chmod +x "$VENV/bin/sgl-model-gateway"
  done_stage 02-wheels-install
fi
if need 02b-fa3shim; then
  echo "[stage2b] FA3 interface shim"
  sp=$(python -c "import site;print(site.getsitepackages()[0])")
  dl "https://raw.githubusercontent.com/Dao-AILab/flash-attention/fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89/hopper/flash_attn_interface.py" \
     "$sp/flash_attn_3/flash_attn_interface.py"
  done_stage 02b-fa3shim
fi

# --- Stage 3: Megatron-LM (public radixark) + bridges + DSA patch ------------
if need 03-megatron; then
  echo "[stage3] mbridge + Megatron-LM(radixark, DSA patch) + Megatron-Bridge"
  S=$ROOT/src; mkdir -p "$S"
  gitclone https://github.com/ISEEKYAN/mbridge.git "$S/mbridge"
  (cd "$S/mbridge" && git checkout -q 89eb10887887bc74853f89a4de258c0702932a1c && $PIP --no-deps .)
  gitclone https://github.com/radixark/Megatron-LM.git "$ROOT/Megatron-LM" --recursive -b miles-main --depth 1
  (cd "$ROOT/Megatron-LM" && $PIP -e . --no-deps)
  # PATCH: wire "dsa" into the experimental-attention dispatcher (miles-main ships
  # get_dsa_module_spec_for_backend but only dispatches gated_delta_net).
  _dsaf="$ROOT/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py"
  if [ -f "$_dsaf" ] && ! grep -q '"dsa"' "$_dsaf"; then
    python3 - "$_dsaf" <<'PYEOF'
import sys
p=sys.argv[1]; s=open(p).read()
old='    if config.experimental_attention_variant == "gated_delta_net":\n        return get_gated_delta_net_module_spec(config=config, backend=backend)\n    else:'
new='    if config.experimental_attention_variant == "gated_delta_net":\n        return get_gated_delta_net_module_spec(config=config, backend=backend)\n    elif config.experimental_attention_variant == "dsa":\n        return get_dsa_module_spec_for_backend(config=config, backend=backend)\n    else:'
assert old in s, "dispatcher block not found (upstream changed?)"
open(p,"w").write(s.replace(old,new,1)); print("patched dsa dispatch")
PYEOF
  fi
  gitclone https://github.com/radixark/Megatron-Bridge.git "$S/Megatron-Bridge" -b bridge --depth 1
  (cd "$S/Megatron-Bridge" && $PIP --no-deps --no-build-isolation .)
  gitclone https://github.com/fzyzcjy/torch_memory_saver.git "$S/torch_memory_saver"
  (cd "$S/torch_memory_saver" && git checkout -q d64a639 && $PIP --no-cache-dir --no-deps .)
  $PIP --no-deps megatron-energon multi-storage-client
  $PIP --no-deps --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"
  $PIP --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 torchvision==0.26.0+cu130
  done_stage 03-megatron
fi

# --- Stage 4: requirements + numpy pin + cudnn -------------------------------
if need 04-requirements; then
  echo "[stage4] miles requirements + cudnn + numpy<2 + sglang import-deps + mooncake"
  $PIP -r "$ROOT/miles/requirements.txt"
  $PIP nvidia-cudnn-cu13==9.16.0.29
  $PIP "numpy<2" "scipy==1.13.1"
  $PIP --index-url https://download.pytorch.org/whl/cu130 --no-deps torchvision
  $PIP IPython openai anthropic einops fastapi orjson msgspec interegular llguidance \
       modelscope partial_json_parser prometheus-client py-spy pyzmq python-multipart \
       setproctitle sentencepiece tiktoken gguf distro easydict compressed-tensors \
       openai-harmony soundfile aiohttp requests psutil pydantic nvidia-ml-py uvicorn
  $PIP mooncake-transfer-engine
  # modelopt was installed --no-deps (so it can't move torch); pulp is its LP-solver
  # dep, imported by modelopt.torch.opt.searcher via the bridge -> install it here.
  $PIP pulp
  done_stage 04-requirements
fi

# --- Stage 4c: fast-hadamard-transform (GLM-5.2 DSA indexer) -----------------
if need 04c-fast-hadamard; then
  echo "[stage4c] fast-hadamard-transform (cu13 build)"
  $PIP wheel setuptools ninja
  # Install each separately: bundling a failing pkg made the whole line fail
  # under `|| true`, silently leaving nvcc uninstalled. nvidia-cuda-nvcc (cu13,
  # v13.3) lands nvcc at nvidia/cu13/bin/nvcc.
  $PIP nvidia-cuda-nvcc || true
  $PIP nvidia-cuda-cccl || true
  $PIP nvidia-cuda-runtime-cu13 || true
  # nvidia is a PEP-420 namespace package -> __file__ is None; use __path__[0].
  CU13=$(python -c "import os,nvidia;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
  (cd "$CU13/lib" && for f in libcudart libcublas libcublasLt; do [ -e "$f.so" ] || ln -sf "$(ls $f.so.* | head -1)" "$f.so"; done)
  (cd "$CU13" && [ -e lib64 ] || ln -sf lib lib64)
  gitclone https://github.com/Dao-AILab/fast-hadamard-transform.git "$ROOT/src/fast-hadamard-transform"
  (cd "$ROOT/src/fast-hadamard-transform" && git checkout -q e7706faf8d1c3b9f241e36860640ad1dac644ede)
  CUDA_HOME="$CU13" PATH="$CU13/bin:$PATH" TORCH_CUDA_ARCH_LIST="10.0" MAX_JOBS=16 \
    $PIP --no-build-isolation --no-deps "$ROOT/src/fast-hadamard-transform"
  done_stage 04c-fast-hadamard
fi

# --- Stage 4d: tilelang (REQUIRED: glm5.py hard-imports the DSA kernels) ------
# abi3 wheel (cp38-abi3 works on py3.12); tvm_ffi is its runtime dep. Cached in
# the shared uv cache from a prior install; this pins the same 0.1.x line.
if need 04d-tilelang; then
  echo "[stage4d] tilelang (+ tvm_ffi) for the GLM-5.2 DSA kernels"
  $PIP "apache-tvm-ffi==0.1.12" || $PIP apache-tvm-ffi || true
  $PIP tilelang || $PIP tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
  python -c "import tilelang; print('tilelang', tilelang.__version__)"
  done_stage 04d-tilelang
fi

# --- Stage 5: sglang (import-time dep of miles train_async) ------------------
if need 05-sglang; then
  echo "[stage5] sglang (sglang-miles) + sgl_kernel"
  S=$ROOT/src; mkdir -p "$S"
  export CARGO_HOME=/workspace/.cache/cargo RUSTUP_HOME=/workspace/.cache/rustup
  if ! command -v rustc >/dev/null 2>&1 && [ ! -x "$CARGO_HOME/bin/rustc" ]; then
    curl --proto '=https' --tlsv1.2 --retry 6 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
  fi
  export PATH="$CARGO_HOME/bin:$PATH"
  gitclone https://github.com/sgl-project/sglang.git "$S/sglang" -b sglang-miles --depth 1
  $PIP --no-deps sglang-kernel==0.4.3
  (cd "$S/sglang" && $PIP -e "python" --no-deps)
  done_stage 05-sglang
fi

# --- Stage 6: miles editable (already cloned: radixark/miles) ----------------
if need 06-miles; then
  echo "[stage6] miles editable"
  (cd "$ROOT/miles" && $PIP -e . --no-deps)
  done_stage 06-miles
fi

# --- Stage 7: apply overlays (my own files that live inside the clones) -------
if need 07-overlays; then
  echo "[stage7] apply overlays/ into cloned trees"
  if [ -d "$ROOT/overlays" ]; then
    # each path under overlays/ mirrors its destination relative to $ROOT
    (cd "$ROOT/overlays" && find . -type f ! -name README.md -print0 | \
      while IFS= read -r -d '' f; do
        dest="$ROOT/${f#./}"; mkdir -p "$(dirname "$dest")"; cp "$f" "$dest"
        echo "  overlay -> ${f#./}"
      done)
  fi
  done_stage 07-overlays
fi

echo "[done] verifying imports..."
python -c "import megatron; print('megatron OK')"
python -c "import tilelang; print('tilelang OK')"
python -c "import importlib; importlib.import_module('miles_plugins.models.glm5.glm5'); print('glm5 spec import OK')"
echo "[build.sh] COMPLETE"

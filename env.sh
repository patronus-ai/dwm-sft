#!/usr/bin/env bash
# Runtime shell for zhe-li's independent Miles GLM-5.2 stack.
#   source /workspace/users/zhe-li/repos/glm-sft/env.sh
ROOT=/workspace/users/zhe-li/repos/glm-sft
source /workspace/users/zhe-li/env/activate.sh
source "$ROOT/.venv/bin/activate"

# torch cu130 bundles CUDA-13 libs under site-packages/nvidia/*/lib (+ cu13/lib);
# TE/megatron dlopen libcublas.so.13 etc., so put those on the loader path.
_NV=$(python -c "import os,glob,torch;b=os.path.join(os.path.dirname(torch.__file__),'..','nvidia');print(':'.join(glob.glob(os.path.join(b,'*','lib'))+glob.glob(os.path.join(b,'*','lib','*'))+glob.glob(os.path.join(b,'cu13','lib'))))")
export LD_LIBRARY_PATH="${_NV}:${LD_LIBRARY_PATH:-}"

export MEGATRON_PATH="$ROOT/Megatron-LM"
export PYTHONPATH="$ROOT/Megatron-LM:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GLM-5.2 DSA provider expectations
export INDEXER_ROPE_NEOX_STYLE=0
export NVSHMEM_DISABLE_NCCL=1

# Single-node default: loopback collectives. (Multi-node runs override
# NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME + RoCE knobs; RDMA libs are present
# system-wide, so no vendored libs needed here.)
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

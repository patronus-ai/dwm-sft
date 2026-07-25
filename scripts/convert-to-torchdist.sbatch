#!/usr/bin/env bash
#SBATCH --job-name=zheli-convert-dist
#SBATCH --partition=gpu-high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
#
# One-time HF -> torch_dist conversion (CHECKPOINT CACHE). Subsequent training
# runs `--load` the produced dir and skip the slow bridge conversion.
#   Test:  MODEL_ARGS_SH=...5layer.sh HF_IN=<5layer> SAVE_OUT=<cache> GPUS=1
#   Real:  MODEL_ARGS_SH=...744B.sh   HF_IN=<FP8>     SAVE_OUT=<cache> GPUS=8
set -euo pipefail
ROOT=/workspace/users/zhe-li/repos/glm-sft
source "$ROOT/env.sh"
cd "$ROOT/miles"

GPUS=${GPUS:-8}
MODEL_ARGS_SH=${MODEL_ARGS_SH:?set MODEL_ARGS_SH to the arch script}
HF_IN=${HF_IN:?set HF_IN to the HF checkpoint dir}
SAVE_OUT=${SAVE_OUT:?set SAVE_OUT to the torch_dist cache dir}
TP=${TP:-8}
mkdir -p "$SAVE_OUT" "$ROOT/logs"

source "$MODEL_ARGS_SH"     # MODEL_ARGS (arch)

# torchrun single-node; the converter auto-picks PP=world when PP=1 to shard the
# model, and saves a torch_dist checkpoint that reshards on load.
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 2000 ))
echo "[convert] gpus=$GPUS TP=$TP hf=$HF_IN -> $SAVE_OUT"
torchrun --nproc-per-node="$GPUS" --nnodes=1 \
  --master-addr=127.0.0.1 --master-port="$MASTER_PORT" \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --tensor-model-parallel-size "$TP" \
  --expert-tensor-parallel-size "$TP" \
  --megatron-to-hf-mode bridge \
  --hf-checkpoint "$HF_IN" \
  --save "$SAVE_OUT"
echo "[convert] done rc=$? -> $SAVE_OUT"
ls -la "$SAVE_OUT" | head
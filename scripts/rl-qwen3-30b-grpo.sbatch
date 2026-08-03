#!/usr/bin/env bash
#SBATCH --job-name=zheli-qwen3-grpo
#SBATCH --partition=gpu-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# GRPO of Qwen3-30B-A3B on dapo-math-17k (miles README example), adapted to our env:
# recipe scripts/run_qwen3_30b_a3b.py handles download(idempotent)/convert/ray/sglang/train.
# 1 node B200; paths redirected off the docker /root defaults into our in-scope tree.
set -euo pipefail
ROOT=/workspace/users/zhe-li/repos/glm-sft
source "$ROOT/env.sh" >/dev/null 2>&1
cd "$ROOT/miles"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export MASTER_ADDR=127.0.0.1
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray_zheli_grpo
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray /tmp/ray_zheli_grpo /workspace/users/zhe-li/tmp/ray 2>/dev/null || true
python scripts/run_qwen3_30b_a3b.py \
  --hardware B200 \
  --num-gpus-per-node 8 \
  --num-nodes 1 \
  --model-dir "$ROOT/models-rl" \
  --data-dir "$ROOT/data-rl" \
  --megatron-path "$ROOT/Megatron-LM" \
  --output-dir "$ROOT/out-rl" \
  --extra-env-vars "MILES_HOST_IP=127.0.0.1"

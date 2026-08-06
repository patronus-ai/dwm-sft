#!/usr/bin/env bash
#SBATCH --job-name=zheli-fp8base-20L
#SBATCH --partition=gpu-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=00:40:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# fp8-BASE PORT test: 20L @ 4k, 1 node. GLM_FP8_BASE=1 -> build frozen base as
# blockwise-fp8 (fp8_model_init). Fast iteration to shake out the plumbing:
#  1) does the base build fp8 (mem ~half)?  2) does the load quantize correctly?
#  3) which ops need the bf16 carve-out (kv_up_proj / weights_proj)?
# Baseline (bf16 base) loss on this exact config: 8.36 -> 7.13 -> 5.39 (compare).
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_20layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_20layer
export SEQ_LEN=4096 MAX_TOK=4096
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_4k.parquet
export TP=8 EP=8 ETP=1
export GLM_FP8_BASE=1
export RUN=fp8base-20L-4k
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/memscale-1node.sbatch

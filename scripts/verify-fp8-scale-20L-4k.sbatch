#!/usr/bin/env bash
#SBATCH --job-name=zheli-verify-fp8scale
#SBATCH --partition=gpu-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# Verify the GLM5 fp8 base-scale load fix: 20L @ 4k, 1 node. If the base now loads
# with its blockwise scale, train/loss should be FINITE (was NaN with the garbage base).
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_20layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_20layer
export SEQ_LEN=4096 MAX_TOK=4096
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_4k.parquet
export TP=8 EP=8 ETP=1
export RUN=verify-fp8scale-20L-4k
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/memscale-1node.sbatch

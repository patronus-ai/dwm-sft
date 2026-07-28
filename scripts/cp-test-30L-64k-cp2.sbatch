#!/usr/bin/env bash
#SBATCH --job-name=zheli-cptest-cp2
#SBATCH --partition=gpu-low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# CP=2 test. 30L @ 64k, 2 nodes, identical to the cp1 wrapper except CP.
# Grid TP8/CP2/PP1/EP8/ETP1 (world16: DP1; EP*ETP=8=TP*DP). Each rank: 32k seq (64k/CP2).
# DSA_ALLOW_CP=1 opts into the scoped assert bypass (GLM's plugin path IS CP-aware).
# Expect: runs (past the assert), ~half the per-rank activation vs cp1, matching loss.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export SEQ_LEN=65536
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_64k.parquet
export TP=8 CP=2 PP=1 EP=8 ETP=1 NUM_EPOCH=1
export DSA_ALLOW_CP=1
export WANDB_PROJECT=glm-cptest WANDB_GROUP=cptest RUN=cptest-30L-64k-cp2
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch

#!/usr/bin/env bash
#SBATCH --job-name=zheli-cptest-cp1
#SBATCH --partition=gpu-low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# CP=1 baseline for the CP correctness test. 30L @ 64k, 2 nodes.
# Grid TP8/CP1/PP1/EP16/ETP1 (world16: DP2; EP*ETP=16=TP*DP). Each rank: full 64k seq.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export SEQ_LEN=65536
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_64k.parquet
export TP=8 CP=1 PP=1 EP=16 ETP=1 NUM_EPOCH=1
export WANDB_PROJECT=glm-cptest WANDB_GROUP=cptest RUN=cptest-30L-64k-cp1
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch

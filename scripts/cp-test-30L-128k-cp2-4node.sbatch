#!/usr/bin/env bash
#SBATCH --job-name=zheli-cp2-4node-128k
#SBATCH --partition=gpu-low
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# 30L @ 128k on 4 nodes: CP=2 + DP=2 + EP=16. Grid TP8/CP2/PP1/EP16/ETP1
# (world32: DP2; EP*ETP=16=TP*DP). vs CP=4/EP8 (base ~69 GB, OOM'd in MoE backward
# by ~8.6 GB), this HALVES base/rank to ~36 GB (EP16) -> ~33 GB freed, while keeping
# CP=2 (64k queries/rank) + the chunked indexer (no O(S^2) dense logits).
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export SEQ_LEN=131072
export MAX_TOK=65536   # micro-step token cap (NOT context); halves the 21.56 GiB MoE activation. seq stays 128k.
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_128k.parquet
export TP=8 CP=2 PP=1 EP=16 ETP=1 NUM_EPOCH=1
export DSA_ALLOW_CP=1
export DSA_INDEXER_Q_CHUNK=4096
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=glm-cptest WANDB_GROUP=cptest RUN=cptest-30L-128k-cp2-4node
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch

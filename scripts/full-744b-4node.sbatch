#!/usr/bin/env bash
#SBATCH --job-name=zheli-full744b
#SBATCH --partition=gpu-low
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# FULL GLM-5.2 744B (78 layers) LoRA-SFT on 4 nodes, WITH the base-scale fix.
# Grid TP8/CP1/PP1/EP32/ETP1 (world32, DP4; EP*ETP=32=TP*DP; EP32|256 experts).
# EP32 shards the base to ~50 GB/rank -> fits in bf16 at short context. 128k needs
# the fp8-base port (in progress); this proves the FULL model trains correctly now.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B.sh
export MODEL=/workspace/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
export SEQ_LEN=4096 MAX_TOK=4096
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_4k.parquet
export TP=8 CP=1 PP=1 EP=32 ETP=1 NUM_EPOCH=1
export WANDB_PROJECT=glm-full744b WANDB_GROUP=full744b RUN=full744b-4k
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch

#!/usr/bin/env bash
#SBATCH --job-name=zheli-cp2-128k
#SBATCH --partition=gpu-low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# THE HEADLINE TEST: 30L @ FULL 128k on 2 nodes with CP=2.
# Grid TP8/CP2/PP1/EP8/ETP1 (world16: DP1; EP*ETP=8=TP*DP). Each rank: 128k/2 = 64k seq
# -> the exact activation profile that FIT on 1 node at 64k. CP=1 OOMs at 128k on 1/2/4
# nodes; if CP=2 fits here, context parallelism unlocks 128k on the PUBLIC stack.
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B_30layer.sh
export MODEL=/workspace/users/zhe-li/repos/glm-sft/models/GLM-5.2_30layer
export SEQ_LEN=131072
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_128k.parquet
export TP=8 CP=2 PP=1 EP=8 ETP=1 NUM_EPOCH=1
export DSA_ALLOW_CP=1
export WANDB_PROJECT=glm-cptest WANDB_GROUP=cptest RUN=cptest-30L-128k-cp2
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch

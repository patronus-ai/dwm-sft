#!/usr/bin/env bash
#SBATCH --job-name=zheli-full744b-128k-cp2
#SBATCH --partition=gpu-low
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=07:00:00
#SBATCH --output=/workspace/users/zhe-li/repos/glm-sft/logs/slurm-%x-%j.out
# FULL GLM-5.2 744B @ 128k, 20 steps -- CP2/EP16 (ROBUST grid).
# EP32/CP1 fit 4 steps but OOM'd stochastically over 20 (recompute checkpoints
# ~117 GB dominate, <6 GB headroom). CP2 HALVES the checkpoints (~58 GB) and the
# per-rank token spike -- that saving outweighs base 50->99 GB (EP16). Peak ~161 vs
# ~175 GB -> ~17 GB headroom. Same grid the 30-layer 128k runs used successfully.
# Grid TP8/CP2/PP1/EP16/ETP1 (world32, DP2; EP*ETP=16=TP*DP; EP16|256 experts).
export MODEL_ARGS_SH=/workspace/users/zhe-li/repos/glm-sft/miles/scripts/models/glm5.2-744B-A40B.sh
export MODEL=/workspace/.cache/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
export PROMPT_DATA=/workspace/users/zhe-li/repos/glm-sft/data/sft_128k.parquet
export SEQ_LEN=131072 MAX_TOK=65536
export TP=8 CP=2 PP=1 EP=16 ETP=1 NUM_EPOCH=5   # 5 x 4 steps/epoch = 20 steps
export TARGET_MODULES="q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"   # LoRA on ATTENTION only (not experts) -- kills the 10.88 GiB expert-LoRA OOM
export DSA_ALLOW_CP=1 DSA_INDEXER_Q_CHUNK=4096
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export WANDB_PROJECT=glm-full744b WANDB_GROUP=full744b-128k RUN=full744b-128k-cp2-20steps
exec bash /workspace/users/zhe-li/repos/glm-sft/scripts/sft-128k-glm744b.sbatch

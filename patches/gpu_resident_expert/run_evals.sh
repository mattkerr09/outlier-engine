#!/bin/bash
set -e
cd /workspace/outlier-eval
export HF_HOME=/mnt/1tbfin3/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 40B V3.2 MMLU START $(date -u) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m lm_eval \
    --model hf \
    --model_args pretrained=/mnt/1tbfin3/base_40b,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --batch_size 1 \
    --output_path /workspace/outlier-eval/results/40b_v3_2_mmlu_full.json \
    2>&1 | tee /workspace/outlier-eval/logs/eval_40b_final.tee
echo "=== 40B DONE $(date -u) ==="

echo "=== 70B V3.2 MMLU START $(date -u) ==="
CUDA_VISIBLE_DEVICES=1 python3 -m lm_eval \
    --model hf \
    --model_args pretrained=/mnt/1tbfin3/base_70b,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --batch_size 1 \
    --output_path /workspace/outlier-eval/results/70b_v3_2_mmlu_full.json \
    2>&1 | tee /workspace/outlier-eval/logs/eval_70b_final.tee
echo "=== 70B DONE $(date -u) ==="

echo "BOTH_EVALS_COMPLETE" > /workspace/EVAL_DONE

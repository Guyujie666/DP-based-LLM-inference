#!/usr/bin/env bash
set -euo pipefail

dataset=wikitext2
model=openPangu-Embedded-7B-V1.1
soft_token_num=100
target_attack_rates=(0.02 0.1 0.15)
quant_levels=(4)
proj_dim=128
mechanism='ternary'

export ASCEND_RT_VISIBLE_DEVICES=6
export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface

log_dir='./logs_pangu'
mkdir -p "$log_dir"

res_dir='./res_pangu'
mkdir -p "$res_dir"

# 把 attack_rates 拼成 ta0.02_0.1_0.2_0.4_0.6
join_rates() { local IFS="_"; printf "%s" "ta$(printf "%s_" "${target_attack_rates[@]}" | sed 's/_$//')"; }
rates_tag="$(join_rates)"
safe_model="${model//\//__}"

bit="${quant_levels[0]}"  # 若只有一个量化级别
results_out="$res_dir/simcse_eval__${dataset}__${safe_model}__proj${proj_dim}__${mechanism}__bit${bit}__${rates_tag}.json"
log_file="$log_dir/log_${dataset}_bit_${bit}_${mechanism}_proj_${proj_dim}_${rates_tag}_chat_mask_new.txt"

nohup python test.py \
  --model "$model" \
  --model_name_or_path "$model" \
  --dataset "$dataset" \
  --soft \
  --emb_ft \
  --dp \
  --soft_token_num "$soft_token_num" \
  --emb_ckpt /data/gyj/pangu/openPangu-Embedded-7B-V1.1/output/pangu_adamw_lr0.0002_steps20000_proj_dim128/c4_no_norm_mask_0.05_0.5/projs.pth \
  --proj_dim "$proj_dim" \
  --noise_mechanism Ternary \
  --mechanism_name ternary \
  --quant_levels "${quant_levels[@]}" \
  --attack_rates "${target_attack_rates[@]}" \
  --out_file_root ./output_pangu \
  --optimizer adamw \
  --prompt_lr 0.001 \
  --max_steps 4000 \
  --clip_bound 0.05 \
  --search_json_dir ./pangu_clean \
  --results_out "$results_out" \
  --gptq_models microsoft/DialoGPT-medium KoalaAI/OPT-1.3b-Chat \
  > "$log_file" 2>&1 &



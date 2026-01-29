dataset=wikitext2
model=openPangu-Embedded-7B-V1.1
# model=Qwen/Qwen2.5-7B
soft_token_num=100
target_attack_rates=(0.02 0.1 0.15)
quant_levels=(4)
proj_dim=128
mechanism='ternary'   # 机制名与搜索阶段保持一致（文件名里也是这个）
clip_bound=0.05

export ASCEND_RT_VISIBLE_DEVICES=7
# export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface

# 1) 先搜索（你原来的命令不变）
python auto_search_dp.py \
  --dataset $dataset \
  --base_model "$model" \
  --privacy_mechanisms $mechanism \
  --quant_levels ${quant_levels[@]} \
  --target_attack_rates ${target_attack_rates[@]} \
  --proj_dim $proj_dim \
  --emb_ckpt /data/gyj/pangu/openPangu-Embedded-7B-V1.1/output/pangu_adamw_lr0.0002_steps20000_proj_dim128/c4_no_norm_mask_0.05_0.5/projs.pth \
  --subset_size 2000 \
  --output_dir ./pangu_clean

wait

# # 2) 目录准备
out_file='./output_pangu'
mkdir -p "$out_file"
log_dir='./log_ternary_pangu'
mkdir -p "$log_dir"

# 3) 训练：按 quant_level 取 μ 列表，再按目标攻击率逐一跑
for opt in adamw; do
for lr in 0.001; do
for steps in 4000; do
for bit in "${quant_levels[@]}"; do

    # === 关键一步：查表得到与 target_attack_rates 一一对应的 mu 列表 ===
    mus=($(python lookup_mu.py \
        --output_dir ./pangu_clean \
        --dataset "$dataset" \
        --model "$model" \
        --proj_dim "$proj_dim" \
        --mechanism "$mechanism" \
        --quant_level "$bit" \
        --targets "${target_attack_rates[@]}"))

    # 安全检查
    if [ ${#mus[@]} -ne ${#target_attack_rates[@]} ]; then
        echo "[ERR] lookup_mu 返回数量与 target_attack_rates 不一致" >&2
        exit 1
    fi
    # dataset=c4

    # 逐一组合：同一索引下的 ta 和 mu 对应
    for idx in "${!target_attack_rates[@]}"; do
        ta="${target_attack_rates[$idx]}"
        mu="${mus[$idx]}"

        echo "[RUN] quant=${bit}  target_attack_rate=${ta}  mu=${mu}"

        save_dir="$out_file/${opt}_lr${lr}_steps${steps}_proj_dim${proj_dim}/${dataset}_transfer_${bit}_bit/ta_${ta}"
        mkdir -p "$save_dir"

        nohup python emb_proj_soft_ft.py \
            --model "$model" \
            --dataset "$dataset" \
            --eval_every_steps 500 \
            --proj_dim "${proj_dim}" \
            --prompt_lr "${lr}" \
            --soft_token_num "${soft_token_num}" \
            --mu "${mu}" \
            --max_steps "${steps}" \
            --optimizer "${opt}" \
            --emb_ckpt /data/gyj/pangu/openPangu-Embedded-7B-V1.1/output/pangu_adamw_lr0.0002_steps20000_proj_dim128/c4_no_norm_mask_0.05_0.5/projs.pth \
            --output_dir "$save_dir" \
            --noise_mechanism 'Ternary' \
            --per_device_train_batch_size 2 \
            --model_name_or_path "$model" \
            --soft \
            --clip_bound $clip_bound \
            --prefix_ratio_min 0.05 \
            --prefix_ratio_max 0.5 \
            --dp \
            --quant_level "$bit" \
            > "$log_dir/log_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}_bit_${bit}_ternary_mu_${mu}_proj_${proj_dim}_ta_${ta}.txt" 2>&1 &

        wait
    done

done
done
done
done

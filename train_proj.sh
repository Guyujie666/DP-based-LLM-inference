dataset=c4
# model=meta-llama/Llama-2-7b-hf
# model=baffo32/decapoda-research-llama-7b-hf
# model=meta-llama/Llama-3.1-8B
model=pangu/openPangu-Embedded-7B-V1.1
# model=Qwen/Qwen2.5-7B
proj_dims=(128)
# mus=(6 8 10 12 14 16 18 20 22 24 26 28 30)
mus=(20)
chips=(0)
bits=(32)
soft_token_num=100
#########lr 0.0001, steps 1000 for emb and 0.001,10000 for soft###############
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface
export HF_TRUST_REMOTE_CODE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
out_file='./output'
if [ ! -d "$out_file" ]; then
    echo "Creating directory $out_file"
    mkdir -p "$out_file"
fi
log_file='./log_proj'
if [ ! -d "$log_file" ]; then
    echo "Creating directory $log_file"
    mkdir -p "$log_file"
fi
for opt in adamw; do
for lr in 0.0002; do
for steps in 20000; do
for bit in ${bits[@]}; do
for mu in ${mus[@]}; do
for chi in ${chips[@]}; do
for proj_dim in ${proj_dims[@]}; do

ASCEND_RT_VISIBLE_DEVICES=4 nohup python emb_proj_soft_ft.py \
    --model ${model} \
    --dataset ${dataset} \
    --eval_every_steps 1000 \
    --seqlen 1024 \
    --proj_dim ${proj_dim} \
    --prompt_lr ${lr} \
    --max_steps ${steps} \
    --optimizer ${opt} \
    --prefix_ratio_min 0.05 \
    --prefix_ratio_max 0.5 \
    --output_dir ${out_file}/pangu_${opt}_lr${lr}_steps${steps}_proj_dim${proj_dim}/${dataset}_no_norm_mask_0.05_0.5_new \
    --per_device_train_batch_size 2 --model_name_or_path ${model} --emb_ft --root './' 2>&1 > ${log_file}/log_gptq_${opt}_lr${lr}_${dataset}_steps${steps}_proj${proj_dim}_pangu_no_norm_new_mask_0.05_0.5.txt &

wait
done
done
done
done 
done
done
done
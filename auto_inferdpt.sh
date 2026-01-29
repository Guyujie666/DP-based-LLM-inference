
mc=0
# export CUDA_VISIBLE_DEVICES='0'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface
export TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
log_file='./log_inferdpt_pangu'
if [ ! -d "$log_file" ]; then
    echo "Creating directory $log_file"
    mkdir -p "$log_file"
fi



dataset=wikitext2
model=openPangu-Embedded-7B-V1.1

echo "===> Step 1: export vocab for ${model}"
(
    cd inferdpt || exit 1
    python export_vocab.py --model "${model}"
)
echo "===> export_vocab 完成"
echo "===> Step 2: run auto_inferdpt_evaluate.py"

ASCEND_RT_VISIBLE_DEVICES=5 nohup python auto_inferdpt_evaluate.py \
    --model_name_or_path ${model} \
    --hf_tokenizer ${model} \
    --dataset ${dataset} \
    --gptq_model microsoft/DialoGPT-medium KoalaAI/OPT-1.3b-Chat \
    --targets 0.02 0.1 0.15 --mc $mc 2>&1 > $log_file/inferdpt_evaluate_${model##*/}_${dataset}_mc_${mc}_chat1.txt &


set -e  # 一旦有命令出错(返回值非0)就立刻退出，可选但推荐

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface
export PANGU_PATH=/data/gyj
# python download.py

echo "===> Running train_proj.sh ..."
bash train_proj.sh

# echo "===> Running train_soft.sh ..."
# bash train_soft.sh

# echo "===> Running test_cse.sh ..."
# bash test_cse.sh
# echo "===> Test of our method completed. See results in ./res_pangu/"

# echo "===> Running auto_inferdpt.sh ..."
# bash auto_inferdpt.sh
# echo "===> Inference with DPT completed. See results in ./results_mc0/"

# echo "All done!"

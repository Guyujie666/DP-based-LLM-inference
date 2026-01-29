import torch
import numpy as np
import argparse

import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    LlamaForCausalLM,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    OPTForCausalLM,GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
)

from datasets import load_dataset
import matplotlib.pyplot as plt
# import seaborn as sns
from torch.utils.data import DataLoader
import torch.nn.functional as F
# ==== NEW: 可能用到的损失函数
from torch.nn import CrossEntropyLoss
from dp_noise import *
# import os
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    PeftModel
)
import torch.nn as nn
import os, json, time
from pathlib import Path
PANGU_PATH = os.getenv("PANGU_PATH", "/default/pangu/path")

PREFIX_TEXT = "Please continue the text\n\n"
CONTROL_TEXT = "\no_think"
PRE_OUTPUT_MARKERS = ["[unused16]", "[unused16]", "[unused17]"]  # 回答起始三标记
POST_OUTPUT_MARKER  = "[unused10]"                               # 闭合标记

def ensure_special_tokens(tokenizer, model=None):
    """确保 [unused16],[unused17],[unused10] 存在；若新增词表则可选 resize 模型嵌入"""
    add_tokens = [t for t in ["[unused16]", "[unused17]", "[unused10]"] if t not in tokenizer.get_vocab()]
    if add_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        if model is not None and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
    return {
        "u16": tokenizer.convert_tokens_to_ids("[unused16]"),
        "u17": tokenizer.convert_tokens_to_ids("[unused17]"),
        "u10": tokenizer.convert_tokens_to_ids("[unused10]"),
    }

def build_infer_input(prompt: str, tokenizer):
    """
    构造推理输入（prompt/问题 + \no_think + 三标记）：
    PREFIX_TEXT + prompt + CONTROL_TEXT + [u16][u16][u17]
    """
    pre_markers = "".join(PRE_OUTPUT_MARKERS)
    full_text = f"{PREFIX_TEXT}{prompt}{CONTROL_TEXT}{pre_markers}"
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    return enc, full_text

def extract_formal_answer(decoded_full: str):
    """从完整解码文本中，提取 [u16][u16][u17] 与 [u10] 之间的正式回答（不含标记）。"""
    pre = "".join(PRE_OUTPUT_MARKERS)
    post = POST_OUTPUT_MARKER
    try:
        s = decoded_full.index(pre) + len(pre)
        e = decoded_full.index(post, s)
        return decoded_full[s:e]
    except ValueError:
        return ""

def cut_after_eos_id(generated_ids: torch.LongTensor, eos_id: int):
    """在 token 级别截取到第一个 eos_id（含），仅处理 batch=1。"""
    arr = generated_ids.tolist()
    if eos_id in arr:
        pos = arr.index(eos_id)
        return generated_ids[:pos+1]
    return generated_ids



def _str_key(x):
    if isinstance(x, float):
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return str(x)

def _safe_tag(s: str) -> str:
    s = s.strip().replace("\\", "/")
    for ch in ["/", ":", " ", "@", "#", "?", "&", "=", "+"]:
        s = s.replace(ch, "__")
    return s

def load_mu_from_search(output_dir, dataset, model_name, proj_dim, mechanism, quant_level, attack_rate):
    safe = _safe_tag(model_name)
    fp = Path(output_dir) / f"{dataset}__{safe}__proj{proj_dim}__{mechanism.lower()}.json"
    if not fp.exists():
        return None
    with open(fp, "r") as f:
        payload = json.load(f)
    qkey = _str_key(quant_level); tkey = _str_key(attack_rate)
    row = payload.get("table", {}).get(qkey, {}).get(tkey)
    if not row or str(row.get("param_name","")).lower() != "mu":
        return None
    return float(row["param_value"])

def find_soft_ckpt_under(out_file_root, optimizer, prompt_lr, max_steps, proj_dim,
                         dataset, quant_level, attack_rate, mu_val):
    """
    目录规则：
      {out_file_root}/{opt}_lr{lr}_steps{steps}_proj_dim{D}/{dataset}_transfer_{bit}_bit/ta_{attack_rate}/
    优先 *best.pt，否则取该目录下时间最新的 .pt
    """
    bit = int(quant_level)
    ta_key = _str_key(attack_rate)
    leaf = f"{optimizer}_lr{prompt_lr}_steps{max_steps}_proj_dim{proj_dim}"
    base = Path(out_file_root) / leaf / f"{dataset}_transfer_{bit}_bit" / f"ta_{ta_key}"
    print('base path:', base,'mu_val:', mu_val)
    if not base.exists():
        return None
    return os.path.join(base, f"mu_{mu_val}_best.pt")
    # # 优先 best
    # bests = sorted(base.glob("*best.pt"))
    # if bests:
    #     return str(bests[-1])
    # cands = sorted(base.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    # return str(cands[-1]) if cands else None
def find_soft_ckpt_for_c4_transfer(out_file_root, optimizer, prompt_lr, max_steps, proj_dim,
                         dataset, quant_level, attack_rate, mu_val, soft_num):
    """
    目录规则：
      {out_file_root}/{opt}_lr{lr}_steps{steps}_proj_dim{D}/{dataset}_transfer_{bit}_bit/ta_{attack_rate}/
    优先 *best.pt，否则取该目录下时间最新的 .pt
    """
    bit = int(quant_level)
    ta_key = _str_key(attack_rate)
    leaf = f"{optimizer}_lr{prompt_lr}_steps{max_steps}_proj_dim{proj_dim}"
    base = Path(out_file_root) / leaf / f"{dataset}_transfer_{bit}_bit" / f"ta_{ta_key}_soft{soft_num}"
    if not base.exists():
        return None
    return os.path.join(base, f"mu_{mu_val}_best.pt")

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', type=str, default= "decapoda-research/llama-7b-hf")
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--soft_ckpt', type=str, default=None)
parser.add_argument("--hard", action='store_true', default=False)
parser.add_argument("--turn-to-token", action='store_true', default=False,)
parser.add_argument("--fixed", action='store_true', default=False)
parser.add_argument('--max-samples', type=int, default= 200)
parser.add_argument("--soft", action='store_true', default=False)
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/baseline/ptb/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/adamw_lr0.001_steps30000/c4/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/unpruned/ptb/best.ckpt")
parser.add_argument('--sce_model_name', type = str, default = "princeton-nlp/sup-simcse-bert-base-uncased")
parser.add_argument("--clip_bound", type=float, default=1.0,
                    help="GDP level")
parser.add_argument("--dp-rounds", type=int, default=1,
                    help="round of dp")
parser.add_argument('--dataset', type = str, default = "wikitext2")
parser.add_argument('--dtype', type = str, default = "bfloat16")
parser.add_argument('--soft_token_num', type = int, default = 50)
parser.add_argument("--quant_level", type=int,default=32) 
parser.add_argument("--dp", action='store_true', default=False)
parser.add_argument("--mu", type=float, default=10000,
                    help="GDP level")
parser.add_argument("--sparsity", type=float, default=1.0,
                    help="sparsity ratio")
parser.add_argument("--noise_mechanism", type=str, default="Gaussian",
                    choices=["Gaussian", "ChiDP", "Ternary"],)
parser.add_argument("--device", type=str, default="cuda:0", 
                    choices = ["cuda", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    help = "cpu or gpu")
parser.add_argument("--inferdpt", action='store_true', default=False)
parser.add_argument('--gptq_model', type=str, default= "unsloth/llama-2-7b-bnb-4bit")
parser.add_argument("--emb_ft", action='store_true', default=False)
parser.add_argument("--proj_dim", type=int, default=512)
parser.add_argument("--emb_ckpt", type=str, default=None)
# ==== NEW: PPL 评估批大小
parser.add_argument("--ppl_batch_size", type=int, default=8)

######xinzeng
parser.add_argument("--gptq", action="store_true", default=False, help="whether to do GPTQ denoising & eval")

parser.add_argument("--attack_rates", type=float, nargs="+", default=None,
                    help="需要评估的一组攻击成功率，会按规则自动定位 soft prompt ckpt")
parser.add_argument("--quant_levels", type=int, nargs="+", default=None,
                    help="若同时评估多个 bit，请传入多个 quant levels")
parser.add_argument("--out_file_root", type=str, default="/data/dp_soft_prompt/search_results/output_baffo",
                    help="训练 soft prompt 的根目录（包含 {opt}_lr{lr}_steps{steps}_proj_dim{D}/...）")
parser.add_argument("--optimizer", type=str, default="adamw")
parser.add_argument("--prompt_lr", type=str, default="0.001")
parser.add_argument("--max_steps", type=str, default="4000")
parser.add_argument("--mechanism_name", type=str, default="ternary",
                    help="对应 auto_search_dp 写盘的机制名；用于查 μ")
parser.add_argument("--search_json_dir", type=str, default="/data/dp_soft_prompt/search_results",
                    help="auto_search_dp.py 输出 JSON 所在目录，用于查 mu")
parser.add_argument("--results_out", type=str, default=None,
                    help="评估结果的 JSON（追加合并）")
parser.add_argument("--gptq_models", type=str, nargs="+", default=None,
                    help="可选：一组 GPTQ 模型名，逐个评估并记录 SimCSE")
parser.add_argument("--c4_transfer", action="store_true", default=False, help="whether to do C4 transfer")

args = parser.parse_args()

def load_gptq_any(name: str):
    """
    通用加载：按名称启发式判断（与你脚本里 inferdpt 的分支保持一致）
    返回 (model, tokenizer, tag)
    """
    tag = name
    # if 'llama' in name.lower():

    #     gptq_tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    #     gptq_model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, load_in_8bit=True, device_map="auto")
    # # elif 'gpt2' in name.lower():
    # #     gptq_tok = GPT2Tokenizer.from_pretrained(name)
    # #     gptq_model = GPT2LMHeadModel.from_pretrained(name)
    # # elif 'opt' in name.lower():
    # #     gptq_tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    # #     gptq_model = OPTForCausalLM.from_pretrained(name)
    # else:
    gptq_tok = AutoTokenizer.from_pretrained(name)
    if 'llama' in name.lower():
        gptq_model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    else:
        gptq_model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
    gptq_model.eval()
    return gptq_model, gptq_tok, tag

def denoise_with_gptq_list(prompts, generated, gptq_models: list, max_new_tokens=200):
    """
    针对多 gptq 模型分别做“生成后去噪”，返回 {model_tag: denoised_texts}
    """
    results = {}
    for model_name in gptq_models:
        print(f"[DENOISE] Using GPTQ: {model_name}")
        gptq_model, gptq_tok, tag = load_gptq_any(model_name)
        gptq_model.cuda()
        denoised = []
        for i, (p, g) in enumerate(zip(prompts, generated)):
            prompt_template = (
                "Your task is to extend the \"Prefix Text\". Use the \"Perturbed Generation\" as your primary "
                "writing material for your extension. Extract coherent and consistent text from the \"Perturbed "
                "Generation\" and integrate them into your continuation. Ensure a seamless alignment with the "
                "context established by the \"Prefix Text\". Provide only your \"Extended Text\"\n"
                f"——\"Prefix Text\": {p}\n"
                f"——\"Perturbed Generation\": {g}\n"
                "——\"Extended Text\": "
            )
            inputs = gptq_tok(prompt_template, return_tensors="pt").to(gptq_model.device)
            with torch.no_grad():
                out = gptq_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', None),
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=gptq_tok.eos_token_id
                )
            txt = gptq_tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            denoised.append(txt)
        results[tag] = denoised
        torch.cuda.empty_cache()
    return results

def load_simcse_model(model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer
def _is_first_step(past_key_values, cache_position):
    # 老接口：None 仍然表示第一步
    print('past_key_values', past_key_values)
    print('cache_position', cache_position)
    if past_key_values is None and cache_position is None:
        return True
    # 新接口：past_key_values 是 Cache 对象，第一步长度为 0
    try:
        # 仅当是新版 Cache 对象时可用
        from transformers.cache_utils import Cache
        if isinstance(past_key_values, Cache):
            # 有的版本是 get_seq_length()，有的版本是 .seqlen 或 .seen_tokens
            if hasattr(past_key_values, "get_seq_length"):
                return past_key_values.get_seq_length() == 0
            if hasattr(past_key_values, "seqlen"):
                return past_key_values.seqlen == 0
            if hasattr(past_key_values, "seen_tokens"):
                return past_key_values.seen_tokens == 0
    except Exception:
        pass
    # 兜底：根据 cache_position 的起始位置判断（首步一般从 0 开始）
    if cache_position is not None:
        # cache_position 可能是 1D 或 2D，取最小值判断
        return int(cache_position.min().item()) == 0
    return False
class DPEmbeddingFTModel(torch.nn.Module):
    def __init__(self, peft_model, dp_args):
        """
        带有嵌入微调和差分隐私噪声的模型
        
        参数:
            peft_model: PEFT模型
            dp_args: 包含差分隐私参数的对象
        """
        super().__init__()
        self.peft_model = peft_model
        self.dp_args = dp_args
        
        # 获取基础模型的配置和嵌入层
        if hasattr(peft_model, 'get_base_model'):
            self.base_model = peft_model.get_base_model()
        else:
            # 如果是原始模型，直接使用
            self.base_model = peft_model
        self.config = self.base_model.config
        self.embedding_layer = self.base_model.get_input_embeddings()
        
        # 初始化投影和反投影层
        hidden_size = self.config.hidden_size
        proj_dim = dp_args.proj_dim
        
        # 获取模型的设备和数据类型
        example_param = next(self.base_model.parameters())
        device = example_param.device
        dtype = example_param.dtype
        
        # 创建投影和反投影层
        self.cus_proj = nn.Linear(hidden_size, proj_dim, bias=False).to(device=device, dtype=dtype)
        # if self.dp_args.emb_ft:
        #     self.cus_proj = P.spectral_norm(self.cus_proj, n_power_iterations=3)
        #     print('open sp norm')        
        self.cus_deproj = nn.Linear(proj_dim, hidden_size, bias=False).to(device=device, dtype=dtype)
        
        if dp_args.emb_ckpt is None:
            # 初始化为单位映射
            print('no emb proj')
        else:
            emb_checkpoint = torch.load(dp_args.emb_ckpt, map_location=torch.device("cpu"))
            self.cus_proj.load_state_dict(emb_checkpoint['proj'])
            self.cus_deproj.load_state_dict(emb_checkpoint['deproj'])
            print('load emb ckpt!')
        
        # 保存原始嵌入函数
        self.original_embedding_func = self.embedding_layer.forward
        
        # 替换嵌入层的前向传播
        self.embedding_layer.forward = self.emb_ft_forward
        
        # 初始化状态变量
        self.init_token_len = 0
        self.init_embedding = None
        self.first_step = True
        self._dp_apply_next_call = False  # 仅在下一次调用 Embedding.forward 时应用自定义处理
        self.ext_prompt = None  # 可选的扩展提示，用于生成时使用

        
    def emb_ft_forward(self, input_ids):

        original = self.original_embedding_func(input_ids)
        # return original
        # print("is first_step:", self._dp_apply_next_call)
        # print("original embedding shape:", original.shape)

        # 仅当开关开启时，对“首批输入”做自定义处理；之后的自回归步直接走原始嵌入
        if not self._dp_apply_next_call:
            return original

        # === 你的自定义处理逻辑 ===
        if getattr(self.dp_args, 'dp', False):
            if getattr(self.dp_args, 'emb_ft', False):
                x = self.cus_proj(original)
                # 注意：若你在这里做了 F.normalize，会改变噪声校准；确认这是你想要的
                # x = F.normalize(x, p=2, dim=-1)
                x = get_noise_embeddings_for_emb(x, self.dp_args, self.base_model)
                out = self.cus_deproj(x)
            else:
                out = get_noise_embeddings_for_emb(original, self.dp_args, self.base_model)
        else:
            if getattr(self.dp_args, 'emb_ft', False):
                out = self.cus_deproj(self.cus_proj(original))
            else:
                out = original
        if self.ext_prompt is not None:
            ext_embed = self.original_embedding_func(self.ext_prompt)
            out = torch.cat([out, ext_embed], dim=1)

        if getattr(self.dp_args, 'turn_to_token', False):
            B, T, H = out.shape
            W = self.embedding_layer.weight.detach()
            sims = F.normalize(out.float(), p=2, dim=-1).reshape(-1, H) @ \
                F.normalize(W.float(), p=2, dim=-1).t()
            idx = torch.argmax(sims, dim=-1)
            out = W[idx].reshape(B, T, H).to(out.dtype)

        # 关键：只处理一次，立刻关掉
        self._dp_apply_next_call = False
        return out


    
    def _extend_labels(self, labels, ignore_index=-100):
        """扩展标签以匹配prompt长度"""
        if not hasattr(self.dp_args, 'soft') or not self.dp_args.soft:
            return labels
        
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        n_tokens = self.dp_args.soft_token_num if hasattr(self.dp_args, 'soft_token_num') else 20
        
        return torch.cat(
            [
                torch.full((n_batches, n_tokens), ignore_index).to(labels.device),
                labels,
            ],
            dim=1,
        )
    
    def forward(self, **kwargs):
        # 仅在生成期间，才根据缓存判断“是否首步”
        # 透传到 peft 模型
        return self.peft_model(**kwargs)

    
    def generate(self, **kwargs):
        # self._in_generate = True
        # # 保险起见：进入 generate 前，先允许首批处理一次
        self._dp_apply_next_call = True
        # try:
        return self.peft_model.generate(**kwargs)
        # finally:
        #     # 无论成功或异常，都清理状态
        #     self._in_generate = False
        #     self._dp_apply_next_call = False

    
    def __getattr__(self, name):
        """转发未定义的方法到peft_model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.peft_model, name)

def preprocess_dataset(dataset_name, max_samples=100):
    """处理数据集，支持wikitext2和ptb"""
    processed_data = []
    
    if dataset_name == "wikitext2":
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split='test')
        for item in dataset:
            text = item['text']
            if len(text.split()) >= 100:  # 只保留较长的段落
                # print(len(text.split()))
                processed_data.append({
                    'text': text,
                    'prompt': ' '.join(text.split()[:50]),  # 取前20个词作为提示
                    'reference': ' '.join(text.split()[50:])  # 剩余部分作为参考
                })
                if len(processed_data) >= max_samples:
                    break
    
    elif dataset_name == "ptb":
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        for item in dataset:
            text = item['sentence']  # PTB使用'sentence'字段
            if len(text.split()) >= 30:
                processed_data.append({
                    'text': text,
                    'prompt': ' '.join(text.split()[:20]),
                    'reference': ' '.join(text.split()[20:])
                })
                if len(processed_data) >= max_samples:
                    break
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: wikitext2, ptb")
                
    return processed_data

def compute_simcse_similarity(texts1, texts2, simcse_model, simcse_tokenizer, batch_size=8):
    """使用SimCSE计算两组文本之间的语义相似度"""
    simcse_model.eval()
    device = next(simcse_model.parameters()).device
    
    similarities = []
    
    for i in tqdm(range(0, len(texts1), batch_size), desc="Computing similarities"):
        batch_texts1 = texts1[i:i+batch_size]
        batch_texts2 = texts2[i:i+batch_size]
        
        # 编码第一组文本
        inputs1 = simcse_tokenizer(batch_texts1, padding=True, truncation=True, return_tensors="pt")
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        with torch.no_grad():
            outputs1 = simcse_model(**inputs1)
            embeddings1 = outputs1.last_hidden_state[:, 0]  # 取[CLS]标记的表示
            embeddings1 = F.normalize(embeddings1, p=2, dim=1)  # 归一化
        
        # 编码第二组文本
        inputs2 = simcse_tokenizer(batch_texts2, padding=True, truncation=True, return_tensors="pt")
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        with torch.no_grad():
            outputs2 = simcse_model(**inputs2)
            embeddings2 = outputs2.last_hidden_state[:, 0]  # 取[CLS]标记的表示
            embeddings2 = F.normalize(embeddings2, p=2, dim=1)  # 归一化
        
        # 计算余弦相似度
        batch_similarities = torch.bmm(
            embeddings1.unsqueeze(1), 
            embeddings2.unsqueeze(2)
        ).squeeze().cpu().numpy()
        
        similarities.extend(batch_similarities.tolist())
    
    return similarities

def evaluate_generation_with_simcse(dataset_name, prompts, references, generated_texts, simcse_model, simcse_tokenizer):
    """评估生成文本与参考文本的SimCSE相似度"""
    similarities = compute_simcse_similarity(generated_texts, references, simcse_model, simcse_tokenizer)
    
    avg_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    print(f"\n----- SimCSE Evaluation Results for {dataset_name} -----")
    print(f"Average similarity: {avg_sim:.4f}")
    print(f"Median similarity: {median_sim:.4f}")
    print(f"Min similarity: {min_sim:.4f}")
    print(f"Max similarity: {max_sim:.4f}")
    
    return {
        'avg_similarity': avg_sim,
        'median_similarity': median_sim,
        'min_similarity': min_sim,
        'max_similarity': max_sim,
        'all_similarities': similarities
    }

def generate_with_llama(model, tokenizer, prompts, args, refs, max_length=200):
    """
    协议版生成：
      输入：PREFIX_TEXT + prompt + \no_think + [u16][u16][u17]
      截止：eos_token_id = [unused10]
      输出：返回“正式回答”列表（三标记与 [u10] 之间的内容）
    """
    generated_formal = []
    if args.dataset == 'ptb':
        max_length = 50

    # 取闭合符 id
    eos_id = tokenizer.convert_tokens_to_ids(POST_OUTPUT_MARKER)
    if eos_id is None or eos_id < 0:
        # 兜底（正常不会发生，因为我们在加载后调用了 ensure_special_tokens）
        _ids = ensure_special_tokens(tokenizer, model)
        eos_id = _ids["u10"]

    for i, user_prefix in enumerate(tqdm(prompts, desc="Generating texts")):
        # 1) 构造输入：prompt/问题 + \no_think + 三标记
        enc, full_in = build_infer_input(user_prefix, tokenizer)
        input_ids = enc["input_ids"].to(model.device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(model.device)

        # 2) 生成：遇到 [unused10] 停止
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.eos_token_id
            )[0]

        # 3) 仅取新增部分，并按 eos 再次截断（稳妥）
        gen_only = out_ids[input_ids.shape[1]:]
        gen_only = cut_after_eos_id(gen_only, eos_id)

        # 4) 提取正式回答（从三标记到 [unused10] 之间）
        decoded_full = tokenizer.decode(gen_only, skip_special_tokens=False)
        formal = extract_formal_answer(decoded_full)
        generated_formal.append(decoded_full)

        # 可选打印前 5 条
        if i < 5:
            raw_text = tokenizer.decode(gen_only, skip_special_tokens=False)
            print(f"[{i+1}] Prompt : {user_prefix!r}")
            print(f"[{i+1}] RawGen : {raw_text!r}")
            print(f"[{i+1}] Formal: {formal!r}")
            print(f"[{i+1}] Ref   : {refs[i] if i < len(refs) else ''!r}")
            print("-" * 60)

    # 与你原有评测接口一致：返回一个“生成文本”列表
    return generated_formal


# ==== NEW: 条件 PPL 的数据集与 collate（只对 continuation 计 loss）
class CondPPLDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, prompts, continuations, max_length=1024, only_continuation=True):
        self.tok = tokenizer
        self.prompts = prompts
        self.conts = continuations
        self.max_length = max_length
        self.only_cont = only_continuation

        # 兜底 pad_token
        if getattr(self.tok, "pad_token_id", None) is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        p_ids = self.tok(self.prompts[i], add_special_tokens=False, return_tensors="pt").input_ids[0]
        c_ids = self.tok(self.conts[i],   add_special_tokens=False, return_tensors="pt").input_ids[0]

        input_ids = torch.cat([p_ids, c_ids], dim=0)  # [Lp + Lc]
        if input_ids.size(0) > self.max_length:
            # 左截断（保留右侧最近的 tokens）
            input_ids = input_ids[-self.max_length:]
            cont_len = min(c_ids.size(0), input_ids.size(0))
            prompt_len = input_ids.size(0) - cont_len
        else:
            prompt_len = p_ids.size(0)
            cont_len = c_ids.size(0)

        labels = input_ids.clone()
        if self.only_cont:
            labels[:prompt_len] = -100  # 只对 continuation 计 loss

        attn_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

def cond_ppl_collate(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels    = torch.nn.utils.rnn.pad_sequence([b["labels"]    for b in batch], batch_first=True, padding_value=-100)
    attn_mask = torch.nn.utils.rnn.pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

def append_results_json(path: str, rows: list):
    """
    结果追加合并：若存在则读旧 rows 去重后合并再原子写回
    row 结构建议：
      {
        "time": "...", "dataset": ..., "base_model": ..., "mechanism": ..., "proj_dim": ...,
        "quant_level": ..., "attack_rate": ..., "mu": ..., "soft_ckpt": "...",
        "simcse_avg_baseline": float,
        "simcse_avg_per_gptq": {"modelA": float, ...}
      }
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if dst.exists():
        try:
            with open(dst, "r") as f:
                obj = json.load(f)
                old = obj.get("rows", [])
        except Exception as e:
            print(f"[WARN] read results_out failed: {e}, will recreate")

    # 用 (dataset, base_model, proj_dim, quant_level, attack_rate, mu, soft_ckpt) 做唯一键
    def key(r):
        return (
            r.get("dataset"), r.get("base_model"), r.get("proj_dim"),
            r.get("quant_level"), _str_key(r.get("attack_rate")), _str_key(r.get("mu")),
            r.get("soft_ckpt")
        )
    m = {key(r): r for r in old}
    for r in rows:
        k = key(r)
        if k not in m:
            m[k] = r  # 只追加，不覆盖

    new_rows = list(m.values())
    tmp = dst.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump({"rows": new_rows}, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, dst)
    print(f"[OK] results appended → {dst}")

def evaluate_one_setting(args, soft_ckpt_path, mu_value, quant_level, tag_suffix="",model=None):
    """
    载入 soft_ckpt + mu，生成一轮，然后：
      - 计算 baseline SimCSE
      - 若提供 gptq_models：对每个做去噪 → 再计算 SimCSE
    返回一条或多条结果（list[dict]），以及 prompts/refs/generated（便于可选复用）
    """
    # 1) 重载 soft prompt（保持你现有 DPEmbeddingFTModel 逻辑）
    base_model = model  # 复用已加载的大模型/PEFT包装
    if args.soft and soft_ckpt_path is not None:
        print(f"[LOAD SOFT] {soft_ckpt_path}")
        sd = torch.load(soft_ckpt_path, map_location="cpu")
        # 你的脚本里：model.peft_model.prompt_encoder.load_state_dict(...)
        base_model.peft_model.prompt_encoder.load_state_dict(sd, strict=False)

    # 更新 mu / 量化
    args.mu = float(mu_value)
    args.quant_level = int(quant_level)

    # 2) 预处理数据 & 生成
    processed = preprocess_dataset(args.dataset, max_samples=args.max_samples)
    prompts = [x["prompt"] for x in processed]
    refs    = [x["reference"] for x in processed]
    max_length = 200 if args.dataset != 'ptb' else 50

    generated = generate_with_llama(base_model, tokenizer, prompts, args, refs, max_length=max_length)

    # 3) baseline SimCSE
    print(f"[EVAL] Baseline SimCSE evaluation")
    base_res = evaluate_generation_with_simcse(args.dataset, prompts, refs, generated, simcse_model, simcse_tokenizer)
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "base_model": args.model_name_or_path,
        "mechanism": args.noise_mechanism,
        "proj_dim": args.proj_dim,
        "quant_level": int(quant_level),
        "attack_rate": None,  # 稍后填
        "mu": float(mu_value),
        "soft_ckpt": soft_ckpt_path,
        "simcse_avg_baseline": float(base_res["avg_similarity"]),
        "simcse_avg_per_gptq": {},
        "tag": tag_suffix,
    }

    # 4) 多 gptq 去噪评估（可选）
    if args.gptq_models:
        denoised_map = denoise_with_gptq_list(prompts, generated, args.gptq_models, max_length)
        for mname, den_texts in denoised_map.items():
            mres = evaluate_generation_with_simcse(args.dataset, prompts, refs, den_texts, simcse_model, simcse_tokenizer)
            row["simcse_avg_per_gptq"][mname] = float(mres["avg_similarity"])

    return row

def evaluate_one_setting_temp(args, soft_ckpt_path, mu_value, quant_level, tag_suffix="",model=None):
    """
    载入 soft_ckpt + mu，生成一轮，然后：
      - 计算 baseline SimCSE
      - 若提供 gptq_models：对每个做去噪 → 再计算 SimCSE
    返回一条或多条结果（list[dict]），以及 prompts/refs/generated（便于可选复用）
    """
    # 1) 重载 soft prompt（保持你现有 DPEmbeddingFTModel 逻辑）
    base_model = model  # 复用已加载的大模型/PEFT包装
    # if args.soft and soft_ckpt_path is not None:
    #     print(f"[LOAD SOFT] {soft_ckpt_path}")
    #     sd = torch.load(soft_ckpt_path, map_location="cpu")
    #     # 你的脚本里：model.peft_model.prompt_encoder.load_state_dict(...)
    #     base_model.peft_model.prompt_encoder.load_state_dict(sd, strict=False)

    # 更新 mu / 量化
    # args.mu = float(mu_value)
    # args.quant_level = int(quant_level)
    args.dp = False  # 评测 PPL 时不加噪声

    # 2) 预处理数据 & 生成
    processed = preprocess_dataset(args.dataset, max_samples=args.max_samples)
    prompts = [x["prompt"] for x in processed]
    refs    = [x["reference"] for x in processed]
    max_length = 200 if args.dataset != 'ptb' else 50

    generated = generate_with_llama(base_model, tokenizer, prompts, args, refs, max_length=max_length)

    # 3) baseline SimCSE
    print(f"[EVAL] Baseline SimCSE evaluation")
    base_res = evaluate_generation_with_simcse(args.dataset, prompts, refs, generated, simcse_model, simcse_tokenizer)
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "base_model": args.model_name_or_path,
        "mechanism": args.noise_mechanism,
        "proj_dim": args.proj_dim,
        "quant_level": int(quant_level),
        "attack_rate": None,  # 稍后填
        "mu": float(mu_value),
        "soft_ckpt": soft_ckpt_path,
        "simcse_avg_baseline": float(base_res["avg_similarity"]),
        "simcse_avg_per_gptq": {},
        "tag": tag_suffix,
    }

    # 4) 多 gptq 去噪评估（可选）
    if args.gptq_models:
        denoised_map = denoise_with_gptq_list(prompts, generated, args.gptq_models, max_length)
        for mname, den_texts in denoised_map.items():
            mres = evaluate_generation_with_simcse(args.dataset, prompts, refs, den_texts, simcse_model, simcse_tokenizer)
            row["simcse_avg_per_gptq"][mname] = float(mres["avg_similarity"])

    return row

# ==== NEW: PPL 评估（严格按有效 token 计）
@torch.no_grad()
def evaluate_ppl(model, val_loader, use_shift=False):
    """
    - use_shift=False：走 HF labels= 路径（推荐）
    - use_shift=True ：手动 shift + CrossEntropyLoss(ignore_index=-100)
    返回：平均 PPL（所有 batch 合并）
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    if use_shift:
        loss_fct = CrossEntropyLoss(ignore_index=-100)

    for batch in tqdm(val_loader, desc="Computing PPL"):
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        labels    = batch["labels"].cuda(non_blocking=True)
        attn_mask = batch["attention_mask"].cuda(non_blocking=True)

        if not use_shift:
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss  # 有效 token 上的平均 CE
            valid = (labels != -100).sum().item()
            total_nll    += (loss.float().item() * valid)
            total_tokens += valid
        else:
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            valid = (shift_labels != -100).sum().item()
            total_nll    += (loss.float().item() * valid)
            total_tokens += valid

    ppl = float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1))))
    return ppl

# ==== 主流程
if args.dtype == 'auto':
    dtype = 'auto'
elif args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
else:
    raise NotImplementedError

if 'pangu' in args.model_name_or_path.lower():
    args.model_local_path = f"{PANGU_PATH}/{args.model_name_or_path}"


    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_local_path, 
        use_fast=False, 
        trust_remote_code=True,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_local_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cuda",
        local_files_only=True
    )
    device_id = torch.npu.current_device()
    props = torch.npu.get_device_properties(device_id)

    # 型号名（例如 Ascend 910B 等）
    name = props.name

    # 总显存（以 GB 为单位）
    total_mem_gb = props.total_memory / (1024**3)

    print(f"NPU device id: {device_id}")
    print(f"NPU name     : {name}")
    print(f"Total memory : {total_mem_gb:.2f} GB")
    print(f"Loading: {PANGU_PATH}/{args.model_local_path}")
    print(model.device, model)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Tokenizer value: {tokenizer}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16
    )
if args.soft:
    print("Loading soft prompt model...")

    init_method = PromptTuningInit.RANDOM
    prompt_text = None

    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=init_method,
        num_virtual_tokens=args.soft_token_num,
    )

    # 创建PEFT模型
    peft_model = get_peft_model(model, peft_config)
    model = DPEmbeddingFTModel(peft_model, args)
    if args.soft_ckpt is not None:
        print(f"Loading soft prompt from {args.soft_ckpt}...")
        state_dicts = torch.load(args.soft_ckpt)
        soft_prompt_state_dict = state_dicts
        model.peft_model.prompt_encoder.load_state_dict(soft_prompt_state_dict, strict=False)
else:
    model = DPEmbeddingFTModel(model, args)

# ==== NEW: 兜底 pad_token，便于 batch pad
# if getattr(tokenizer, "pad_token_id", None) is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.model_max_length = getattr(model.config, "max_position_embeddings", 4096)

# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.eos_token_id
# model.generation_config.pad_token_id = tokenizer.eos_token_id
# model.seqlen = 1024
# model.cuda()
model.eval()

simcse_model, simcse_tokenizer = load_simcse_model(args.sce_model_name)
simcse_model = simcse_model.to(args.device)

# row = evaluate_one_setting_temp(
#                 args, soft_ckpt_path=None, mu_value=100000, quant_level=4, tag_suffix=f"temp",model=model
#             )

######新增
if args.attack_rates:
    # 如果没传 quant_levels，就用当前 args.quant_level；否则多 bit 循环
    qlevels = args.quant_levels if args.quant_levels else [args.quant_level]
    collected = []

    for ql in qlevels:
        for ta in args.attack_rates:
            # 1) 自动找 ckpt


            # 2) 自动查 mu（来自 auto_search_dp 的 JSON）
            if args.c4_transfer:
                mu_val = load_mu_from_search(
                    output_dir=args.search_json_dir,
                    dataset='wikitext2',
                    model_name=args.model_name_or_path,
                    proj_dim=args.proj_dim,
                    mechanism=args.mechanism_name,
                    quant_level=ql,
                    attack_rate=ta
                )
                if mu_val is None:
                    print(f"[WARN] mu NOT found in search json: q={ql}, ta={ta}")
                    continue
                soft_path = find_soft_ckpt_for_c4_transfer(
                    out_file_root=args.out_file_root,
                    optimizer=args.optimizer,
                    prompt_lr=str(args.prompt_lr),
                    max_steps=str(args.max_steps),
                    proj_dim=args.proj_dim,
                    dataset='c4',
                    quant_level=ql,
                    attack_rate=ta,mu_val=mu_val, soft_num=args.soft_token_num
                )
            else:
                mu_val = load_mu_from_search(
                    output_dir=args.search_json_dir,
                    dataset=args.dataset,
                    model_name=args.model_name_or_path,
                    proj_dim=args.proj_dim,
                    mechanism=args.mechanism_name,
                    quant_level=ql,
                    attack_rate=ta
                )
                if mu_val is None:
                    print(f"[WARN] mu NOT found in search json: q={ql}, ta={ta}")
                    continue
                soft_path = find_soft_ckpt_under(
                    out_file_root=args.out_file_root,
                    optimizer=args.optimizer,
                    prompt_lr=str(args.prompt_lr),
                    max_steps=str(args.max_steps),
                    proj_dim=args.proj_dim,
                    dataset=args.dataset,
                    quant_level=ql,
                    attack_rate=ta,mu_val=mu_val
                )
            if soft_path is None:
                print(f"[WARN] soft_ckpt NOT found: q={ql}, ta={ta}")
                continue

            print(f"[EVAL] q={ql}  ta={ta} | mu={mu_val} | ckpt={soft_path}")

            # 3) 评估一次（baseline + 可选多个 GPTQ）
            row = evaluate_one_setting(
                args, soft_ckpt_path=soft_path, mu_value=mu_val,
                quant_level=ql, tag_suffix=f"ta_{_str_key(ta)}",model=model
            )
            row["attack_rate"] = float(ta)
            collected.append(row)

    # 4) 结果落盘（可选）
    if args.results_out and collected:
        append_results_json(args.results_out, collected)

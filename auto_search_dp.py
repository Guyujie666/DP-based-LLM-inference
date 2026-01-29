import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import time
from collections import Counter, defaultdict
import math
from torch.distributions import Gamma
import argparse
import json
import os,tempfile
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
PANGU_PATH = os.getenv("PANGU_PATH", "/default/pangu/path")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

clip_dict = {
    'openPangu-Embedded-7B-V1.1': 0.05,

}
def get_pretrained_model(args):
    if args.base_model =="stevhliu/my_awesome_model":
        base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    elif 'pangu' in args.base_model.lower():
        model_local_path = f"{PANGU_PATH}/{args.base_model}"


        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(
        model_local_path, 
        use_fast=False, 
        trust_remote_code=True,
        local_files_only=True
        )
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_local_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda",
            local_files_only=True
        )
    return tokenizer, base_model
def get_token_embedding(token_id, model, args, squeeze=False):
    """get the token embedding given the input ids"""
    with torch.no_grad():
        if args.base_model =="stevhliu/my_awesome_model":
            embeddings = model.distilbert.embeddings.word_embeddings(token_id)
            # embeddings = model.distilbert.embeddings(token_id)
        elif 'llama' in args.base_model or 'pangu' in args.base_model.lower():
            original_device = token_id.device
            embed_layer = model.get_input_embeddings()   # 通用写法
            device = embed_layer.weight.device   # 获取 embedding 权重的 device
            embeddings = embed_layer(token_id.to(device))
            embeddings = embeddings.to(original_device)  # 转回原始 device
        if squeeze:
            embeddings = embeddings.squeeze()
    return embeddings
def get_closest_token(embedding, tokenizer, model, args):
    """Find the word with the closest embedding."""
    closest_token = None
    if 'gpt2' in args.base_model:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    token_ids = [token_id for _, token_id in vocabulary.items()]
    token_ids = torch.tensor(token_ids).to(args.device)
    word_embeddings = get_token_embedding(token_ids, model, args, squeeze=True)
    # word_embeddings = torch.sign(word_embeddings)

    embedding = embedding.unsqueeze(dim=0)
    embedding = embedding.expand(word_embeddings.size())
    # distance = torch.norm(embedding - word_embeddings, 2, dim=1)
    cos_similarity = F.cosine_similarity(embedding, word_embeddings, dim=1)

    # 如果需要距离而不是相似度，可以转换为余弦距离
    distance = 1 - cos_similarity

    # closest_distances, closest_indices = torch.topk(torch.abs(word_embeddings.reshape(-1)), k=100, largest=True)
    # print(f"Closest distances: {closest_distances}")
    closest_idx = distance.argmin()
    closest_token = token_ids[closest_idx]
    # _visualize_embeddings_3d(embedding[0:1], word_embeddings, closest_indices, closest_distances, 
    #                      token_ids, vocabulary, args)
    return closest_token.item()

def sample_noise_Gaussian(d_shape, noise_stddev, device="cpu"):
    noise = torch.normal(mean=0., std=float(noise_stddev), size=d_shape, device=device)
    return noise

def _str_key(x):
    # 将量化位数/攻击率作为字符串键，避免浮点精度问题
    if isinstance(x, float):
        return f"{x:.6f}".rstrip('0').rstrip('.')
    return str(x)
def _merge_mech_table_in_place(dst_table: dict, src_table: dict, mode: str = "append"):
    """
    将 src_table 合并进 dst_table（两者结构相同：
      {quant_level(str): {target_attack_rate(str): record}}
    ）
    mode:
      - "append": 仅补齐缺失组合；若已存在则“保留旧值不变”
      - "overwrite": 覆盖已有组合（可作为可选开关）
      - "skip": 完全跳过已有 quant_level 的组合（仅新增新的 quant_level）
    """
    assert mode in ("append", "overwrite", "skip")
    for qkey, ta_map in src_table.items():
        if qkey not in dst_table:
            dst_table[qkey] = {}
        if mode == "skip" and qkey in dst_table:
            # 整个 quant_level 已存在，跳过
            continue
        for tkey, rec in ta_map.items():
            if mode == "overwrite":
                dst_table[qkey][tkey] = rec
            else:  # append
                if tkey not in dst_table[qkey]:
                    dst_table[qkey][tkey] = rec
                # 已存在则不改动

def build_canonical_table(results: dict) -> dict:
    """
    将 auto_search_privacy_parameters_comprehensive 返回的嵌套 results
    规整为 {mechanism: {quant_level(str): {target_attack_rate(str): record}}}
    record 示例:
      {
        "param_name": "mu",
        "param_value": 12.34,
        "actual_attack_rate": 0.101,
        "actual_success_rate": 0.899,
        "attack_rate_error": 0.001,
        "success_rate_error": 0.001,
        "dp_rounds": 7,
        "quant_level": 4,
        "target_attack_rate": 0.1,
        "target_success_rate": 0.9,
        "search_time": 12.3
      }
    """
    table = {}
    for ta, qdict in results.items():
        for ql, mechdict in qdict.items():
            for mech, rec in mechdict.items():
                mech_tbl = table.setdefault(mech, {})
                qkey = _str_key(ql)
                ta_key = _str_key(ta)
                qtbl = mech_tbl.setdefault(qkey, {})
                if "error" in rec:
                    qtbl[ta_key] = {"error": rec["error"]}
                else:
                    row = {
                        "param_name": rec["parameter_name"],
                        "param_value": rec["best_parameter"],
                        "actual_attack_rate": rec["actual_attack_rate"],
                        "actual_success_rate": rec["actual_success_rate"],
                        "attack_rate_error": rec["attack_rate_difference"],
                        "success_rate_error": rec["success_rate_difference"],
                        "dp_rounds": rec.get("dp_rounds", None),
                        "quant_level": ql,
                        "target_attack_rate": ta,
                        "target_success_rate": rec["target_success_rate"],
                        "search_time": rec.get("search_time", 0.0),
                    }
                    qtbl[ta_key] = row
    return table

def write_results_per_mechanism(
    canonical_table: dict,
    *,
    output_dir: str,
    dataset: str,
    model_name_or_path: str,
    proj_dim: int,
    emb_ckpt: str,
    extra_meta: dict = None,
    merge_mode: str = "append",   # "append" | "overwrite" | "skip"
):
    """
    将规范化结果按机制拆分写入文件，带“增量合并”：
    - 若文件不存在：直接写入
    - 若文件存在：读出后按 merge_mode 合并，仅“append”时不会改旧值
    """

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    safe_model = safe_tag(model_name_or_path)

    for mechanism, new_mech_table in canonical_table.items():
        filename = f"{dataset}__{safe_model}__proj{proj_dim}__{mechanism}.json"
        out_path = out_root / filename

        # 1) 准备 payload（默认只含 new 的 table）
        payload = {
            "dataset": dataset,
            "model": model_name_or_path,
            "proj_dim": proj_dim,
            "mechanism": mechanism,
            "emb_ckpt": emb_ckpt,
            "meta": {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "merge_mode": merge_mode,
            },
            "table": new_mech_table,
        }
        if extra_meta:
            payload["meta"].update(extra_meta)

        # 2) 若已有文件，读取并合并
        if out_path.exists():
            try:
                with open(out_path, "r") as f:
                    old = json.load(f)
                # 元信息：保留最早一次的 dataset/model/proj_dim 等，更新 meta.time
                payload["dataset"] = old.get("dataset", payload["dataset"])
                payload["model"] = old.get("model", payload["model"])
                payload["proj_dim"] = old.get("proj_dim", payload["proj_dim"])
                payload["emb_ckpt"] = old.get("emb_ckpt", payload["emb_ckpt"])
                old_table = old.get("table", {})

                # 合并表
                _merge_mech_table_in_place(old_table, new_mech_table, mode=merge_mode)
                payload["table"] = old_table
            except Exception as e:
                print(f"[WARN] Failed to read/merge existing file {out_path}: {e}. Writing fresh file.")

        # 3) 原子写：先写临时文件，再替换
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".tmp_{filename}.", dir=str(out_root))
        try:
            with os.fdopen(tmp_fd, "w") as tmpf:
                json.dump(payload, tmpf, indent=2, ensure_ascii=False)
                tmpf.flush()
                os.fsync(tmpf.fileno())
            os.replace(tmp_path, out_path)  # POSIX 原子替换
        finally:
            # 若异常，清理临时文件
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

        print(f"[OK] wrote (merge={merge_mode}) → {out_path}")

def safe_tag(s: str) -> str:
    s = s.strip().replace("\\", "/")
    for ch in ["/", ":", " ", "@", "#", "?", "&", "=", "+"]:
        s = s.replace(ch, "__")
    return s
def write_search_json(
    output_dir: str,
    dataset: str,
    model_name_or_path: str,
    proj_dim: int,
    mechanism: str,
    quant_level: int,
    emb_ckpt: str,
    results: list,
    extra_meta: dict = None
):
    """
    results: 列表，每个元素是 {attack_rate, param_value, achieved, iters, status}
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    safe_model = safe_tag(model_name_or_path)
    filename = f"{dataset}__{safe_model}__proj{proj_dim}__{mechanism}_q{quant_level}.json"
    out_path = out_root / filename

    payload = {
        "dataset": dataset,
        "model": model_name_or_path,
        "proj_dim": proj_dim,
        "mechanism": mechanism,
        "quant_level": quant_level,
        "emb_ckpt": emb_ckpt,
        "targets": results,
        "meta": {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    if extra_meta:
        payload["meta"].update(extra_meta)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote search result → {out_path}")

# [Keep all your existing helper functions: sign_flip_noise, quantize_tensor, etc.]
@torch.no_grad()
def build_vocab_matrix(tokenizer, model, args, dtype=torch.float16):
    # 取词表所有 id（注意不同tokenizer的接口）
    if 'gpt2' in args.base_model.lower() or 'pangu' in args.base_model.lower():
        vocab = tokenizer.get_vocab()
        token_ids = torch.tensor([tid for _, tid in vocab.items()], device='cpu', dtype=torch.long)
    else:
        vocab = tokenizer.vocab
        token_ids = torch.tensor([tid for _, tid in vocab.items()], device='cpu', dtype=torch.long)

    # 取 embedding（分批搬到 GPU，避免峰值）
    bs = 8192
    embs = []
    for i in range(0, len(token_ids), bs):
        chunk = token_ids[i:i+bs].to(args.device)
        e = get_token_embedding(chunk, model, args, squeeze=True)  # (bs, H)
        e = F.normalize(e, p=2, dim=1)                              # 先单位化
        embs.append(e.to(dtype))
        torch.cuda.empty_cache()
    vocab_mat = torch.cat(embs, dim=0)                              # (V, H) on GPU
    return token_ids.to(args.device), vocab_mat                      # 归一化且常驻GPU


def init_proj_layers(model, hidden_size, proj_dim, emb_ckpt_path=None, device="cuda", dtype=torch.float32):
    """
    在 model 上挂两个线性层：
      - model.cus_proj:  hidden_size -> proj_dim
      - model.cus_deproj: proj_dim    -> hidden_size
    如提供 emb_ckpt_path，则从 ckpt 加载 state_dict（要求包含 'proj' 和 'deproj'）
    """
    model.cus_proj = nn.Linear(hidden_size, proj_dim, bias=False).to(device=device, dtype=dtype)
    model.cus_deproj = nn.Linear(proj_dim,  hidden_size, bias=False).to(device=device, dtype=dtype)

    if emb_ckpt_path:
        ckpt = torch.load(emb_ckpt_path, map_location="cpu")
        # 兼容：可能直接是两个 state_dict；也可能在 'proj' / 'deproj' 键下
        if isinstance(ckpt, dict) and "proj" in ckpt and "deproj" in ckpt:
            proj_sd = ckpt["proj"]
            deproj_sd = ckpt["deproj"]
        else:
            # 简单兜底：若顶层就是 state_dict
            proj_sd = ckpt.get("proj", ckpt)
            deproj_sd = ckpt.get("deproj", ckpt)
        model.cus_proj.load_state_dict(proj_sd, strict=True)
        model.cus_deproj.load_state_dict(deproj_sd, strict=True)

    # 推理模式，避免梯度与 dropout 等
    model.cus_proj.eval()
    model.cus_deproj.eval()
    return model


@torch.no_grad()
def project_add_noise_deproject(embeds, model, args):
    """
    embeds: [B, T, H] 或 [N, H]
    在投影空间加噪，再反投影回原空间。只改变数值，不改变形状。
    """
    assert hasattr(model, "cus_proj") and hasattr(model, "cus_deproj"), \
        "Projection layers not found. Call init_proj_layers first."

    # 统一到 3D，便于 batch 处理
    squeeze_back = False
    if embeds.dim() == 2:
        embeds = embeds.unsqueeze(0)   # -> [1, N, H]
        squeeze_back = True

    target_device = embeds.device
    if model.cus_proj.weight.device != target_device:
        model.cus_proj = model.cus_proj.to(target_device)
        model.cus_deproj = model.cus_deproj.to(target_device)

    # 投影
    projected = model.cus_proj(embeds)            # [B, T, D]
    # 在投影空间加噪
    if args.noise_type == 'ternary':
        noisy_proj = ternary_noise_encode(projected, args)+ projected
    elif args.noise_type == 'gaussian':
        noisy_proj = sample_noise_Gauss(projected, args) + projected
    elif args.noise_type == 'binary':
        noisy_proj = gauss_binary_noise_encode(projected, args) + projected
    elif args.noise_type in ['chidp', 'chi']:
        noisy_proj = sample_noise_Chi(projected, args) + projected
    else:
        raise ValueError(f"Unknown noise type for projection-space noise: {args.noise_type}")

    # （可选）若你只想对投影后的“增量”做量化，再反投影：
    if args.noise_type in ['gaussian', 'chidp'] and getattr(args, "quant_level", 32) != 32:
        noisy_proj = quantize_tensor(noisy_proj, args.quant_level)

    # 反投影
    deproj = model.cus_deproj(noisy_proj)         # [B, T, H]

    if squeeze_back:
        deproj = deproj.squeeze(0)                # 还原到 [N, H]
    return deproj


def sign_flip_noise(vector, flip_ratio, args):
    noise_mask = torch.bernoulli(
            torch.full(vector.shape, flip_ratio, device=args.device)
        ).bool()
    return vector * (1 - 2 * noise_mask)  # 翻转选中的位
def quantize_tensor(tensor, num_bits):
    """
    随机量化函数 - 实现无偏的随机量化 (Stochastic Quantization)
    
    Args:
        tensor: 输入张量
        num_bits: 量化位数
    
    Returns:
        quantized_tensor: 随机量化后的张量
    """
    # 找到输入张量的最小值和最大值
    # min_val, max_val = tensor.min(dim=-1, keepdim=True)[0], tensor.max(dim=-1, keepdim=True)[0]
    min_val, max_val = torch.min(tensor), torch.max(tensor)

    # 计算量化级别的数量
    q_levels = 2 ** num_bits

    # 计算缩放比例
    scale = (max_val - min_val) / (q_levels - 1)
    
    # 避免除零错误
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # 将连续值映射到 [0, q_levels-1] 范围
    normalized = (tensor - min_val) / scale
    
    # 随机量化：基于小数部分进行概率性舍入
    floor_vals = torch.floor(normalized)
    frac_vals = normalized - floor_vals
    
    # 生成随机数，如果随机数小于小数部分，则向上舍入，否则向下舍入
    random_vals = torch.rand_like(frac_vals)
    quantized = torch.where(random_vals < frac_vals, 
                           floor_vals + 1, 
                           floor_vals)
    
    # 确保量化值在有效范围内
    quantized = quantized.clamp(0, q_levels - 1)
    
    # 还原为原始范围的浮点数
    quantized_tensor = quantized * scale + min_val

    return quantized_tensor

# def quantize_tensor(tensor, num_bits):
#     # 找到输入张量的最小值和最大值
#     min_val, max_val = tensor.min(dim=1, keepdim=True)[0], tensor.max(dim=1, keepdim=True)[0]

#     # 计算量化级别的数量
#     q_levels = 2 ** num_bits

#     # 计算缩放比例
#     scale = (max_val - min_val) / (q_levels - 1)

#     # 量化到整数，然后还原为浮点数
#     quantized = ((tensor - min_val) / scale).round().clamp(0, q_levels - 1)
#     quantized_tensor = quantized * scale + min_val

#     return quantized_tensor
def sample_noise_Chi(init_emb, args):
    size= init_emb.shape
    device = args.device
    eta = args.eta
    alpha = torch.ones(*size) * size[-1]
    beta = torch.ones(*size) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample()
    # v_lst = -2 * torch.rand(*size) + 1
    v_lst = torch.randn(size)
    v_lst = v_lst / torch.norm(v_lst, dim=-1, keepdim=True)
    noise = l_lst * v_lst
    noise = noise.to(device)
    return noise

def sample_noise_Gauss(init_emb, args):

    noise_std = 2*args.clip_c_bound/args.mu*math.sqrt(init_emb.shape[-1]) #################记得改回去
    # print(f"noise_std: {noise_std}")
    # print('shape of init_emb:', init_emb.shape)
    noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
    ####sparsity
    # random_variable = torch.rand_like(init_emb)
    # noises = torch.where(random_variable <= 1 - args.sparsity, -init_emb, noises)
    # # noise_std = args.train_noise_std if mode == "train" else args.test_noise_std
    # noise_std = 2 * 0.267/mu
    # # print("noise_std:", noise_std)
    # # print("noise_std:", noise_std)
    # noises = sample_noise_Gaussian(init_emb.shape, noise_std, device)

    return noises
def ternary_noise_encode(init_emb, args):
    encoder_list = []
    for i in range(args.dp_rounds):
        mu_ = math.sqrt(args.mu ** 2 / args.dp_rounds)/math.sqrt(init_emb.shape[-1])
        A = math.sqrt(args.sparsity * (4 / mu_ ** 2 + 1) * args.clip_c_bound ** 2)  # 此处看一下init_emb有没有batch维度
        B = A / args.sparsity
        random_variable = torch.rand_like(init_emb)
        ones_tensor = B * torch.ones_like(init_emb)
        zeros_tensor = torch.zeros_like(init_emb)
        encoded_tensor = torch.where(random_variable <= (1 / 2 + init_emb / (2 * A)), ones_tensor, -ones_tensor)
        random_variable = torch.rand_like(encoded_tensor)
        encoded_tensor = torch.where(random_variable <= 1 - A / B, zeros_tensor, encoded_tensor)
        encoder_list.append(encoded_tensor)
        # A = torch.sqrt(args.sparsity * (4/args.mu ** 2 + 1)*torch.max(init_emb)**2)#此处看一下init_emb有没有batch维度
        # B = A/args.sparsity
        # random_variable = torch.rand_like(init_emb)
        # ones_tensor = B*torch.ones_like(init_emb)
        # zeros_tensor = torch.zeros_like(init_emb)
        # encoded_tensor = torch.where(random_variable <= (1 / 2 + init_emb / (2 * A)), ones_tensor, -ones_tensor)
        # random_variable = torch.rand_like(encoded_tensor)
        # encoded_tensor = torch.where(random_variable <= 1 - A / B, zeros_tensor, encoded_tensor)
    stacked_tensors = torch.stack(encoder_list)
    encoded_tensor = torch.mean(stacked_tensors, dim=0)
    noises = encoded_tensor - init_emb
    return noises

def gauss_binary_noise_encode(init_emb, args):
    encoder_list = []
    for i in range(args.dp_rounds):
        mu_ = math.sqrt(args.mu ** 2 / args.dp_rounds)
        noise_std = 2*args.clip_bound/mu_
        noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
        encoded = init_emb + noises
        sign_noises = torch.sign(encoded)
        encoder_list.append(sign_noises)
    stacked_tensors = torch.stack(encoder_list)
    encoded_tensor = torch.mean(stacked_tensors, dim=0)
    noises = encoded_tensor - init_emb
    return noises
class GenericTokenDataset(Dataset):
    """
    将文本数据集（wikitext2 / ptb）转成“干净的 token 序列”，用于最近邻还原评估。
    - 过滤空行、仅取有效分词（<vocab_size 且 >3）
    - 可随机下采样 subset_size 个 token
    """
    def __init__(self, tokenizer, dataset_name: str, subset_size: int = None, seed: int = 42):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.seed = seed

        texts = self._load_raw_texts()
        all_tokens = []
        for txt in tqdm(texts, desc=f"Processing {self.dataset_name}"):
            if not txt or not str(txt).strip():
                continue
            toks = tokenizer.encode(str(txt), add_special_tokens=False)
            valid = [t for t in toks if (t < tokenizer.vocab_size and t > 3)]
            all_tokens.extend(valid)

        # 随机下采样
        if subset_size and len(all_tokens) > subset_size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(all_tokens), subset_size, replace=False)
            all_tokens = [all_tokens[i] for i in idx]

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def _load_raw_texts(self):
        if self.dataset_name in ["wikitext2", "wikitext-2", "wikitext"]:
            ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
            return ds["text"]

        if self.dataset_name in ["ptb", "penn-treebank", "ptb_text_only"]:
            # HF 上常用：ptb_text_only 配置的字段名是 'sentence'
            # 有些镜像可能直接用默认 config；做个兜底
            try:
                ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
                col = "sentence"
            except Exception:
                ds = load_dataset("ptb_text_only", split="test")
                # 兜底找第一个 string 列
                col = next((k for k, v in ds.features.items() if getattr(v, "dtype", None) == "string"), "sentence")
            return ds[col]

        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def make_eval_dataloader(tokenizer, args):
    """
    根据 args.dataset 构造 DataLoader
    """
    ds = GenericTokenDataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        subset_size=args.subset_size or args.test_size  # 与 test_size 对齐
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return ds, loader


@torch.no_grad()
def batch_get_closest_token(embeddings, tokenizer, model, args, batch_size=256,
                            token_ids_all=None, vocab_mat=None, vocab_chunk=16384):
    device = args.device
    N, H = embeddings.shape
    # 归一化并对齐 dtype
    X = F.normalize(embeddings.to(device), p=2, dim=1).to(vocab_mat.dtype)

    # 用 -inf 或该 dtype 的最小有限值初始化
    winners_idx = torch.empty(N, dtype=torch.long, device=device)
    winners_val = torch.full((N,), float('-inf'), dtype=vocab_mat.dtype, device=device)
    # 或者：
    # winners_val = torch.full((N,), torch.finfo(vocab_mat.dtype).min,
    #                          dtype=vocab_mat.dtype, device=device)

    V = vocab_mat.size(0)
    for s in range(0, V, vocab_chunk):
        e = min(s + vocab_chunk, V)
        W = vocab_mat[s:e]  # (chunk, H)

        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            # 在 fp32 做 matmul，结果再 cast 回 vocab_mat.dtype（推荐）
            sims = (X[i:j].to(torch.float32) @ W.to(torch.float32).t()).to(vocab_mat.dtype)
            best_val, best_col = sims.max(dim=1)
            update = best_val > winners_val[i:j]
            winners_val[i:j] = torch.where(update, best_val, winners_val[i:j])
            winners_idx[i:j] = torch.where(update, best_col + s, winners_idx[i:j])

        del W
        torch.cuda.empty_cache()

    return token_ids_all[winners_idx].tolist()



def parallel_noise_generation(embeddings, args, num_workers=None):
    """并行生成噪声"""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    def generate_noise_chunk(embedding_chunk):
        if args.noise_type == 'ternary':
            return ternary_noise_encode(embedding_chunk, args)
        elif args.noise_type == 'gaussian':
            return sample_noise_Gauss(embedding_chunk, args)
        elif args.noise_type == 'binary':
            return gauss_binary_noise_encode(embedding_chunk, args)
        elif args.noise_type == 'chi':
            return sample_noise_Chi(embedding_chunk, args)
        else:
            raise ValueError(f"Unknown noise type: {args.noise_type}")
    
    # 分块处理
    chunk_size = len(embeddings) // num_workers
    chunks = [embeddings[i:i+chunk_size] for i in range(0, len(embeddings), chunk_size)]
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        noise_chunks = list(executor.map(generate_noise_chunk, chunks))
    
    return torch.cat(noise_chunks, dim=0)

def evaluate_defense_success_rate(tokenizer, model, test_loader, args, test_size=1000, token_ids_all=None, vocab_mat=None):
    """
    评估给定参数下的防御成功率
    
    Args:
        tokenizer: 预训练的tokenizer
        model: 预训练的模型
        test_loader: 数据加载器
        args: 参数配置
        test_size: 测试样本数量
    
    Returns:
        defense_success_rate: 防御成功率
    """
    print(f"Evaluating with {args.noise_type}, quant_level={args.quant_level}, "
          f"{'mu=' + str(args.mu) if args.noise_type in ['gaussian', 'ternary'] else 'eta=' + str(args.eta)}")
    
    defense_successes = 0
    total_samples = 0
    samples_processed = 0
    
    with torch.no_grad():
        for batch_tokens in test_loader:
            # 检查是否已处理足够的样本
            if samples_processed >= test_size:
                break
                
            batch_tokens = batch_tokens.to(args.device)
            
            # 限制batch大小以不超过test_size
            remaining_samples = test_size - samples_processed
            if len(batch_tokens) > remaining_samples:
                batch_tokens = batch_tokens[:remaining_samples]
            
            # 获取原始嵌入
            original_embeddings = get_token_embedding(batch_tokens, model, args)
            
            # 应用裁剪
            if args.noise_type in ['binary']:
                all_norms = torch.norm(original_embeddings, p=2, dim=-1, keepdim=True)
                scaling_factor = torch.clamp(args.clip_bound / all_norms, max=1.0)
                clipped_embeddings = original_embeddings * scaling_factor
            elif args.noise_type in ['gaussian', 'ternary']:
                clipped_embeddings = torch.clamp(
                    original_embeddings, 
                    min=-args.clip_c_bound, 
                    max=args.clip_c_bound
                )
            else:  # chi
                clipped_embeddings = original_embeddings
            
            # 生成噪声
            if args.noise_type == 'ternary':
                noises = ternary_noise_encode(clipped_embeddings, args)
            elif args.noise_type == 'gaussian':
                noises = sample_noise_Gauss(clipped_embeddings, args)
            elif args.noise_type == 'binary':
                noises = gauss_binary_noise_encode(clipped_embeddings, args)
            elif args.noise_type == 'chidp':
                noises = sample_noise_Chi(clipped_embeddings, args)
            
            # ==== 投影空间加噪：若开启了投影 ====
            if hasattr(model, "cus_proj") and hasattr(model, "cus_deproj"):
                # 注意：这里建议对“裁剪后的向量”进行投影→加噪→反投影
                noisy_embeddings = project_add_noise_deproject(clipped_embeddings, model, args)
            else:
                # ==== 原来的直接在原空间加噪 ====
                noisy_embeddings = clipped_embeddings + noises
                if args.noise_type in ['gaussian', 'chidp'] and args.quant_level != 32:
                    noisy_embeddings = quantize_tensor(noisy_embeddings, args.quant_level)

            
            predicted_tokens = batch_get_closest_token(
                noisy_embeddings, tokenizer, model, args,
                batch_size=256,
                token_ids_all=token_ids_all,
                vocab_mat=vocab_mat,
                vocab_chunk=16384
            )

            
            original_tokens = batch_tokens.cpu().tolist()
            for orig_token, pred_token in zip(original_tokens, predicted_tokens):
                total_samples += 1
                if orig_token != pred_token:
                    defense_successes += 1
            
            samples_processed += len(batch_tokens)
    
    defense_success_rate = defense_successes / total_samples
    return defense_success_rate

def binary_search_privacy_parameter(tokenizer, model, test_loader, args, target_success_rate, 
                                   tolerance=0.02, max_iterations=15, test_size=2000, token_ids_all=None, vocab_mat=None):
    """
    使用二分搜索找到达到目标防御成功率的隐私参数
    
    Args:
        tokenizer: 预训练的tokenizer
        model: 预训练的模型
        test_loader: 数据加载器
        args: 参数配置
        target_success_rate: 目标防御成功率 (0-1)
        tolerance: 容忍误差
        max_iterations: 最大迭代次数
        test_size: 测试样本数量
    
    Returns:
        best_param: 最优参数值
        best_success_rate: 实际达到的成功率
    """
    print(f"\n=== 开始搜索 {args.noise_type} 机制的隐私参数 ===")
    print(f"目标防御成功率: {target_success_rate:.3f}")
    print(f"量化级别: {args.quant_level}")
    
    # 根据不同机制设置搜索范围
    if args.noise_type == 'gaussian':
        param_name = 'mu'
        low, high = 1.0, 100.0  # mu的搜索范围
    elif args.noise_type == 'ternary':
        param_name = 'mu'
        low, high = 1.0, 2000.0  # mu的搜索范围
        # if 'qwen' in args.base_model.lower():
        #     if args.quant_level < 2:
        #         high = 1000.0  # Qwen模型需要更大的mu范围
        #     else:
        #         high = 400.0  # Qwen模型需要更大的mu范围
    elif args.noise_type == 'chidp':
        param_name = 'eta'
        low, high = 10.0, 1000.0  # eta的搜索范围
    else:
        raise ValueError(f"Unsupported noise type: {args.noise_type}")
    
    best_param = None
    best_success_rate = None
    best_diff = float('inf')
    
    print(f"搜索范围: {param_name} ∈ [{low}, {high}]")
    print("-" * 60)
    
    for iteration in range(max_iterations):
        mid = (low + high) / 2.0
        
        # 设置参数
        if param_name == 'mu':
            args.mu = mid
        else:  # eta
            args.eta = mid
        
        # 如果是ternary，还需要设置dp_rounds
        if args.noise_type == 'ternary':
            args.dp_rounds = 2**args.quant_level - 1
        
        # 评估当前参数下的防御成功率（传入已初始化的模型和数据）
        current_success_rate = evaluate_defense_success_rate(
            tokenizer, model, test_loader, args, test_size, token_ids_all, vocab_mat
        )
        
        diff = abs(current_success_rate - target_success_rate)
        
        print(f"迭代 {iteration+1:2d}: {param_name}={mid:6.2f} -> "
              f"成功率={current_success_rate:.4f} (目标={target_success_rate:.4f}, "
              f"差距={diff:.4f})")
        
        # 更新最佳结果
        if diff < best_diff:
            best_diff = diff
            best_param = mid
            best_success_rate = current_success_rate
        
        # 检查是否达到容忍误差
        if diff <= tolerance:
            print(f"✓ 在第 {iteration+1} 次迭代达到目标！")
            break
        
        # 调整搜索范围
        if current_success_rate < target_success_rate:
            # 成功率太低，需要减小参数（减少噪声）
            high = mid
        else:
            # 成功率太高，需要增大参数（增加噪声）
            low = mid
    
    print("-" * 60)
    print(f"搜索完成！")
    print(f"最佳参数: {param_name} = {best_param:.4f}")
    print(f"实际成功率: {best_success_rate:.4f}")
    print(f"与目标差距: {abs(best_success_rate - target_success_rate):.4f}")
    
    return best_param, best_success_rate

def auto_search_privacy_parameters_comprehensive(args, target_attack_rates, quant_levels, privacy_mechanisms, 
                                               tolerance=0.02, max_iterations=15, test_size=2000, batch_size=32):
    """
    全面的自动搜索多种配置下的隐私参数
    
    Args:
        args: 基础参数配置
        target_attack_rates: 目标攻击率列表 (例如 [0.1, 0.2, 0.3])
        quant_levels: 量化级别列表 (例如 [4, 8, 16])
        privacy_mechanisms: 隐私机制列表 (例如 ['gaussian', 'ternary', 'chidp'])
        tolerance: 容忍误差
        max_iterations: 最大迭代次数
        test_size: 测试样本数量
        batch_size: 批处理大小
    
    Returns:
        results: 搜索结果字典
    """
    
    # 确保输入参数都是列表形式
    if not isinstance(target_attack_rates, list):
        target_attack_rates = [target_attack_rates]
    if not isinstance(quant_levels, list):
        quant_levels = [quant_levels]
    if not isinstance(privacy_mechanisms, list):
        privacy_mechanisms = [privacy_mechanisms]
    
    print(f"\n{'='*100}")
    print(f"🚀 全面自动搜索隐私参数")
    print(f"{'='*100}")
    print(f"目标攻击率: {target_attack_rates}")
    print(f"量化级别: {quant_levels}")
    print(f"隐私机制: {privacy_mechanisms}")
    print(f"测试样本数: {test_size}, 批处理大小: {batch_size}")
    
    # ====== 初始化模型 & 词表矩阵 ======
    print("\n📦 正在初始化模型和数据集...")
    tokenizer, model = get_pretrained_model(args)
    token_ids_all, vocab_mat = build_vocab_matrix(tokenizer, model, args, dtype=torch.float16)

    # model = model.to(args.device)
    hidden_size = model.get_input_embeddings().weight.size(1)

    if args.proj_dim is not None and args.emb_ckpt is not None:
        init_proj_layers(
            model,
            hidden_size=hidden_size,
            proj_dim=args.proj_dim,
            emb_ckpt_path=args.emb_ckpt,
            device=args.device,
            dtype=torch.float32 if model.dtype is torch.float32 else model.dtype
        )
        print(f"[proj] enabled: H={hidden_size} -> D={args.proj_dim}, ckpt={args.emb_ckpt}")
    else:
        print("[proj] disabled (proj_dim or emb_ckpt not provided)")

    model.eval()

    # ====== 使用通用数据集封装（支持 wikitext2 / ptb）======
    _, test_loader = make_eval_dataloader(tokenizer, args)
    print(f"✅ 数据集 {args.dataset} 初始化完成！（subset_size={args.subset_size or args.test_size}）")

    
    results = {}
    total_configs = len(target_attack_rates) * len(quant_levels) * len(privacy_mechanisms)
    current_config = 0
    
    for target_attack_rate in target_attack_rates:
        target_success_rate = 1.0 - target_attack_rate
        
        print(f"\n{'='*80}")
        print(f"🎯 目标攻击率: {target_attack_rate:.3f} (防御成功率: {target_success_rate:.3f})")
        print(f"{'='*80}")
        
        results[target_attack_rate] = {}
        
        for quant_level in quant_levels:
            print(f"\n📊 量化级别: {quant_level}")
            results[target_attack_rate][quant_level] = {}
            args.quant_level = quant_level
            
            for mechanism in privacy_mechanisms:
                current_config += 1
                print(f"\n🔒 [{current_config}/{total_configs}] 隐私机制: {mechanism}")
                args.noise_type = mechanism
                

                start_time = time.time()
                
                # 使用已初始化的模型和数据集进行搜索
                best_param, actual_success_rate = binary_search_privacy_parameter(
                    tokenizer, model, test_loader, args, target_success_rate, 
                    tolerance, max_iterations, test_size, token_ids_all, vocab_mat
                )
                
                search_time = time.time() - start_time
                actual_attack_rate = 1.0 - actual_success_rate
                
                results[target_attack_rate][quant_level][mechanism] = {
                    'target_attack_rate': target_attack_rate,
                    'target_success_rate': target_success_rate,
                    'actual_attack_rate': actual_attack_rate,
                    'actual_success_rate': actual_success_rate,
                    'best_parameter': best_param,
                    'parameter_name': 'mu' if mechanism in ['gaussian', 'ternary'] else 'eta',
                    'attack_rate_difference': abs(actual_attack_rate - target_attack_rate),
                    'success_rate_difference': abs(actual_success_rate - target_success_rate),
                    'search_time': search_time
                }
                
                print(f"✅ 搜索成功！用时: {search_time:.2f}秒")
                    

    
    return results

def print_comprehensive_search_summary(results):
    """打印全面搜索结果摘要"""
    print(f"\n{'='*100}")
    print("🎯 全面搜索结果摘要")
    print(f"{'='*100}")
    
    # 统计信息
    total_configs = 0
    successful_configs = 0
    failed_configs = 0
    
    for target_attack_rate, quant_data in results.items():
        print(f"\n🎯 目标攻击率: {target_attack_rate:.3f} (防御成功率: {1.0-target_attack_rate:.3f})")
        print("=" * 80)
        
        for quant_level, mechanisms in quant_data.items():
            print(f"\n📊 量化级别 {quant_level}:")
            print("-" * 60)
            
            for mechanism, result in mechanisms.items():
                total_configs += 1
                
                if 'error' in result:
                    failed_configs += 1
                    print(f"  ❌ {mechanism:10s}: 搜索失败 - {result['error']}")
                else:
                    successful_configs += 1
                    param_name = result['parameter_name']
                    param_value = result['best_parameter']
                    actual_attack_rate = result['actual_attack_rate']
                    target_attack_rate = result['target_attack_rate']
                    attack_diff = result['attack_rate_difference']
                    search_time = result.get('search_time', 0)
                    
                    print(f"  ✅ {mechanism:10s}: {param_name}={param_value:6.2f} -> "
                          f"攻击率={actual_attack_rate:.4f} (目标={target_attack_rate:.4f}, "
                          f"误差={attack_diff:.4f}, 用时={search_time:.1f}s)")
    
    print(f"\n{'='*80}")
    print(f"📈 总体统计:")
    print(f"  总配置数: {total_configs}")
    print(f"  成功配置: {successful_configs} ({successful_configs/total_configs*100:.1f}%)")
    print(f"  失败配置: {failed_configs} ({failed_configs/total_configs*100:.1f}%)")
    print(f"{'='*80}")

def export_results_to_table(results, filename="privacy_search_results.csv"):
    """将搜索结果导出为CSV表格"""
    import pandas as pd
    
    data = []
    for target_attack_rate, quant_data in results.items():
        for quant_level, mechanisms in quant_data.items():
            for mechanism, result in mechanisms.items():
                if 'error' not in result:
                    data.append({
                        'Target_Attack_Rate': target_attack_rate,
                        'Target_Success_Rate': result['target_success_rate'],
                        'Quantization_Level': quant_level,
                        'Privacy_Mechanism': mechanism,
                        'Parameter_Name': result['parameter_name'],
                        'Best_Parameter': result['best_parameter'],
                        'Actual_Attack_Rate': result['actual_attack_rate'],
                        'Actual_Success_Rate': result['actual_success_rate'],
                        'Attack_Rate_Error': result['attack_rate_difference'],
                        'Success_Rate_Error': result['success_rate_difference'],
                        'Search_Time': result.get('search_time', 0)
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"📊 结果已导出到: {filename}")
    return df


def export_results_index(
    canonical_table: dict,
    *,
    output_dir: str,
    dataset: str,
    model_name_or_path: str,
    proj_dim: int,
):
    safe_model = safe_tag(model_name_or_path)
    out_path = Path(output_dir) / f"{dataset}__{safe_model}__proj{proj_dim}__index.json"
    rows = []
    for mech, qtbl in canonical_table.items():
        for qkey, tdict in qtbl.items():
            for tkey, rec in tdict.items():
                r = {"mechanism": mech, "quant_level": qkey, "target_attack_rate": tkey}
                r.update(rec)
                rows.append(r)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_index.", dir=str(Path(output_dir)))
    try:
        with os.fdopen(tmp_fd, "w") as tmpf:
            json.dump({"rows": rows}, tmpf, indent=2, ensure_ascii=False)
            tmpf.flush(); os.fsync(tmpf.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
    print(f"[OK] wrote index → {out_path}")

def str2type(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, torch.dtype):
        return v
    if "float32" in v.lower():
        return torch.float32
    elif "float16" in v.lower():
        return torch.float16

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive privacy parameter auto-search")

    # 基础模型与设备
    parser.add_argument("--base_model", type=str, default="baffo32/decapoda-research-llama-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--emb_ckpt", type=str, default=None)

    # 数据集设置
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb"],
                        help="选择评估数据集：wikitext2 或 ptb (Penn Treebank text-only)")
    parser.add_argument("--subset_size", type=int, default=2000,
                        help="从测试集随机抽样的 token 数量上限，用于搜索评估")

    # 隐私机制与量化
    parser.add_argument("--noise_type", type=str, default="ternary",
                        choices=["gaussian", "ternary", "chidp", "binary"])
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=100.0)
    parser.add_argument("--dp_rounds", type=int, default=1)
    parser.add_argument("--quant_level", type=int, default=4)

    # 裁剪/稀疏
    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--clip_bound", type=float, default=1.0)
    parser.add_argument("--clip_c_bound", type=float, default=None)

    # 搜索设置
    parser.add_argument("--target_attack_rates", type=float, nargs="+",
                        default=[0.02, 0.1, 0.2, 0.4, 0.6])
    parser.add_argument("--quant_levels", type=int, nargs="+", default=[4])
    parser.add_argument("--privacy_mechanisms", type=str, nargs="+", default=["ternary"])
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_precision", type=str2type, default=torch.float32,
                    help = "Precision of base model")

    # 输出
    parser.add_argument("--output_dir", type=str, default="/data/dp_soft_prompt/search_results")

    return parser.parse_args()


def main():
    args = parse_args()

    # 自动填充 clip_c_bound（如你之前所做）
    if getattr(args, "clip_c_bound", None) is None and args.base_model in clip_dict:
        args.clip_c_bound = clip_dict[args.base_model]
    # if args.proj_dim==128 and 'llama' in args.base_model.lower() and args.emb_ckpt is None:
    #     args.clip_c_bound = 0.1
    # 你已有的搜索过程
    print("开始全面自动隐私参数搜索...")
    results = auto_search_privacy_parameters_comprehensive(
        args=args,
        target_attack_rates=args.target_attack_rates,
        quant_levels=args.quant_levels,
        privacy_mechanisms=args.privacy_mechanisms,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        test_size=args.test_size,
        batch_size=args.batch_size
    )

    # 摘要
    print_comprehensive_search_summary(results)

    # 规范化结果为 {mechanism -> quant_level(str) -> target_attack_rate(str) -> row}
    canonical_table = build_canonical_table(results)

    # canonical_table = build_canonical_table(results)
    write_results_per_mechanism(
        canonical_table,
        output_dir=args.output_dir,
        dataset=args.dataset,
        model_name_or_path=args.base_model,
        proj_dim=args.proj_dim,
        emb_ckpt=args.emb_ckpt,
        extra_meta={"quant_levels": args.quant_levels, "targets": args.target_attack_rates},
        merge_mode="overwrite",   # ← 这里控制合并策略
    )
    export_results_index(
        canonical_table,
        output_dir=args.output_dir,
        dataset=args.dataset,
        model_name_or_path=args.base_model,
        proj_dim=args.proj_dim
    )

if __name__ == "__main__":
    # 运行主搜索程序
    main()
    

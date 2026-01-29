
from tqdm import tqdm
import torch
import argparse
import numpy as np
import time
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Union, Optional, List, Dict
import torch.nn as nn
import torch.nn.functional as F
from dp_noise import *
import torch.nn.utils.parametrizations as P

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed, 
    default_data_collator,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from transformers.optimization import Adafactor
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    PeftModel
)
from torch.utils.data import DataLoader
PANGU_PATH = os.getenv("PANGU_PATH", "/default/pangu/path")
os.environ["TRUST_REMOTE_CODE"] = "true"

# ====================== 输出协议常量 ======================
PREFIX_TEXT = "Please continue the text\n\n"   # 输入开头
CONTROL_TEXT = "\no_think"
PRE_OUTPUT_MARKERS = ["[unused16]", "[unused16]", "[unused17]"]  # 输出前缀标记（计入损失）
POST_OUTPUT_MARKER = "[unused10]"                                   # 闭合标记（计入损失）

# ====================== 谱归一化权重烘焙（原样保留） ======================
def bake_spectral_norm_weights(model_state_dict):
    baked_state_dict = {}
    for key, tensor in model_state_dict.items():
        if 'parametrizations.weight.original' in key:
            base_key = key.replace('parametrizations.weight.original', '')
            u_key = base_key + 'parametrizations.weight.0._u'
            v_key = base_key + 'parametrizations.weight.0._v'
            if u_key in model_state_dict and v_key in model_state_dict:
                original_weight = tensor.float()
                u = model_state_dict[u_key].float()
                v = model_state_dict[v_key].float()
                sigma = torch.dot(u, torch.mv(original_weight, v))
                normalized_weight = original_weight / sigma
                clean_key = base_key + 'weight'
                baked_state_dict[clean_key] = normalized_weight.to(tensor.dtype)
                print(f"Baked {key} -> {clean_key}, spectral norm: {torch.linalg.matrix_norm(normalized_weight, ord=2).item():.6f}")
        elif 'parametrizations.weight.0._u' not in key and 'parametrizations.weight.0._v' not in key:
            if 'parametrizations.weight.original' not in key:
                baked_state_dict[key] = tensor
    return baked_state_dict

# ====================== 数据集（插入三标记 + 闭合 + 后缀；严格结构） ======================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, args, mode="train", col_key='text', cutoff=None):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.seqlen = args.seqlen
        
        # 截取子集
        if cutoff is not None:
            dataset = dataset.select(range(min(cutoff, len(dataset))))
        
        # 拼接原文
        if isinstance(col_key, list):
            texts = [" ".join([item[k] for k in col_key]) for item in dataset]
        else:
            texts = [item[col_key] for item in dataset]
        full_text = " ".join(texts)
        
        # 标记化为一长串 ids
        tokenized = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
        self.raw_ids = tokenized.input_ids[0]  # (N,)
        
        # 前缀/后缀 ids
        self.prefix_ids = tokenizer(PREFIX_TEXT, add_special_tokens=False).input_ids
        # self.suffix_ids = tokenizer(SUFFIX_TEXT, add_special_tokens=False).input_ids
        
        # 三个前标记 + 一个闭合标记
        self.pre_marker_ids = [self._tok_id(tok) for tok in PRE_OUTPUT_MARKERS]
        self.post_marker_id = self._tok_id(POST_OUTPUT_MARKER)
        self.control_ids = tokenizer(CONTROL_TEXT, add_special_tokens=False).input_ids
        
        # 每样本需要从原串取多少 core token 才能拼成 seqlen
        self.num_samples = len(self.raw_ids) // self._core_len_per_sample()
    
    def _tok_id(self, tok):
        if tok in self.tokenizer.get_vocab():
            return self.tokenizer.convert_tokens_to_ids(tok)
        return self.tokenizer.encode(tok, add_special_tokens=False)[0]
    
    def _insert_len(self):
        # prefix + control(\no_think) + 3*pre_markers + 1*post_marker
        return len(self.prefix_ids) + len(self.control_ids) + len(self.pre_marker_ids) + 1

    
    def _core_len_per_sample(self):
        core_len = self.seqlen - self._insert_len()
        return max(core_len, 16)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        core_len = self._core_len_per_sample()
        start_idx = idx * core_len
        end_idx = start_idx + core_len
        core = self.raw_ids[start_idx:end_idx].tolist()

        # —— 随机前缀比例（样本内确定性随机）——
        rmin = getattr(self.args, "prefix_ratio_min", 0.3)
        rmax = getattr(self.args, "prefix_ratio_max", 0.9)
        assert 0.0 < rmin <= rmax < 1.0, "prefix_ratio_* 必须在 (0,1) 内且 min<=max"
        rng = np.random.RandomState(int(self.args.seed) ^ int(idx))
        ratio = rng.uniform(rmin, rmax)

        # 切 P|R，保底不小于 8 token
        p_len = int(len(core) * ratio)
        p_len = min(max(p_len, 8), len(core) - 8)
        prefix_ctx = core[:p_len]
        response_part = core[p_len:]

        # 拼接：prefix + P + 三标记 + R + 闭合 + \no_think
        head = self.prefix_ids + prefix_ctx + self.control_ids + self.pre_marker_ids
        tail = [self.post_marker_id]
        final_ids = head + response_part + tail

        # 固定总长；只裁 P→R，绝不裁标记和 \no_think
        max_core_room = self.seqlen - (len(self.prefix_ids) + len(self.pre_marker_ids) + 1 + len(self.control_ids))
        if max_core_room < 16:
            raise ValueError(f"seqlen 太小，至少要容纳特殊标记与后缀，当前余量 {max_core_room}")
        extra = len(final_ids) - self.seqlen
        if extra > 0:
            cut_p = min(extra, len(prefix_ctx))
            prefix_ctx = prefix_ctx[:len(prefix_ctx) - cut_p]
            extra -= cut_p
            if extra > 0:
                response_part = response_part[:max(0, len(response_part) - extra)]
                extra = 0
            final_ids = self.prefix_ids + prefix_ctx + self.pre_marker_ids + response_part + tail

        if len(final_ids) < self.seqlen:
            final_ids = final_ids + [self.tokenizer.pad_token_id] * (self.seqlen - len(final_ids))

        # 结构强校验
        # 校验顺序必须出现： ... control_ids ... u16,u16,u17 ... R ... u10
        seq = final_ids
        # 检查 control 出现在三标记之前
        def _find_subseq(arr, sub):
            for i in range(len(arr)-len(sub)+1):
                if arr[i:i+len(sub)] == sub:
                    return i
            return -1

        ctrl_pos = _find_subseq(seq, self.control_ids)
        # 定位三标记起点
        u16 = self.pre_marker_ids[0]; u17 = self.pre_marker_ids[2]
        triplet_pos = -1
        for i in range(len(seq)-2):
            if seq[i] == u16 and seq[i+1] == u16 and seq[i+2] == u17:
                triplet_pos = i; break

        assert ctrl_pos >= 0, "\\no_think 缺失"
        assert triplet_pos > ctrl_pos, "\\no_think 必须出现在三标记之前"
        assert self.post_marker_id in seq, "闭合标记 [unused10] 缺失"


        return torch.tensor(final_ids, dtype=torch.long)

# ====================== 模型包装（原逻辑 + 掩码元信息注入） ======================
class DPEmbeddingFTModel(torch.nn.Module):
    def __init__(self, peft_model, dp_args):
        super().__init__()
        self.peft_model = peft_model
        self.dp_args = dp_args
        
        if hasattr(peft_model, 'get_base_model'):
            self.base_model = peft_model.get_base_model()
        else:
            self.base_model = peft_model
        self.config = self.base_model.config
        self.embedding_layer = self.base_model.get_input_embeddings()
        
        hidden_size = self.config.hidden_size
        proj_dim = dp_args.proj_dim
        
        example_param = next(self.base_model.parameters())
        device = example_param.device
        dtype = example_param.dtype
        
        self.cus_proj = nn.Linear(hidden_size, proj_dim, bias=False).to(device=device, dtype=dtype)
        if self.dp_args.emb_ft and self.dp_args.sp:
            self.cus_proj = P.spectral_norm(self.cus_proj, n_power_iterations=3)
            print('open sp norm')
        self.cus_deproj = nn.Linear(proj_dim, hidden_size, bias=False).to(device=device, dtype=dtype)
        
        if dp_args.emb_ckpt is None:
            with torch.no_grad():
                nn.init.xavier_uniform_(self.cus_proj.weight)
                nn.init.xavier_uniform_(self.cus_deproj.weight)
                self.cus_proj.weight.normal_(0, 0.02)
                self.cus_deproj.weight.normal_(0, 0.02)
            self.cus_proj.requires_grad_(True)
            self.cus_deproj.requires_grad_(True)
        else:
            emb_checkpoint = torch.load(dp_args.emb_ckpt, map_location=torch.device("cpu"))
            self.cus_proj.load_state_dict(emb_checkpoint['proj'])
            self.cus_deproj.load_state_dict(emb_checkpoint['deproj'])
            print('load emb ckpt!')
        
        self.original_embedding_func = self.embedding_layer.forward
        self.embedding_layer.forward = self.emb_ft_forward
        
        self.init_token_len = 0
        self.init_embedding = None
        self.first_step = True

        # 掩码所需元数据
        self._unused16_id = None
        self._unused17_id = None
        self._unused10_id = None
        self._suffix_ids = None
        self._soft_len = 0
    
    def set_io_format_meta(self, unused16_id, unused17_id, unused10_id, suffix_ids, soft_len):
        self._unused16_id = int(unused16_id)
        self._unused17_id = int(unused17_id)
        self._unused10_id = int(unused10_id)
        self._suffix_ids = list(map(int, suffix_ids)) if suffix_ids is not None else []
        self._soft_len = int(soft_len or 0)

    def set_adv(self, flag: bool):
        self.dp_args.adv = bool(flag)

    def emb_ft_forward(self, input_ids):
        original_embeds = self.original_embedding_func(input_ids)
        if not getattr(self.dp_args, 'dp', False):
            projected = self.cus_proj(original_embeds)
            if self.dp_args.adv:
                deproj = self.cus_deproj(projected + torch.randn_like(projected)*0.01)
            else:
                deproj = self.cus_deproj(projected)
            self.cos_sim = 1 - F.mse_loss(
                original_embeds.float().reshape(-1, original_embeds.shape[-1]),
                deproj.float().reshape(-1, deproj.shape[-1])
            )
            return deproj
        if not self.dp_args.emb_ft and self.dp_args.emb_ckpt is None:
            deproj = get_noise_embeddings(original_embeds, self.dp_args, self.base_model)
        else:
            projected = self.cus_proj(original_embeds)
            projected = get_noise_embeddings_for_emb(projected, self.dp_args, self.base_model)
            deproj    = self.cus_deproj(projected)
        if getattr(self.dp_args, 'turn_to_token', False):
            B, T, H = deproj.shape
            word_emb = self.embedding_layer.weight.detach()
            sims = F.normalize(deproj.float(), p=2, dim=-1).reshape(-1, H) @ \
                F.normalize(word_emb.float(), p=2, dim=-1).t()
            nearest = torch.argmax(sims, dim=-1)
            snapped = word_emb[nearest].reshape(B, T, H).to(deproj.dtype)
            return snapped
        return deproj

    def _extend_labels(self, labels, ignore_index=-100):
        if not hasattr(self.dp_args, 'soft') or not self.dp_args.soft:
            return labels
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]
        n_tokens = self.dp_args.soft_token_num if hasattr(self.dp_args, 'soft_token_num') else 20
        return torch.cat(
            [torch.full((n_batches, n_tokens), ignore_index).to(labels.device), labels],
            dim=1,
        )
    
    def forward(self, **kwargs):
        return self.peft_model(**kwargs)
    
    def generate(self, **kwargs):
        return self.peft_model.generate(**kwargs)
    
    def save_embeddings(self, path, filename="embeddings.pth"):
        torch.save({"proj": self.cus_proj.state_dict(),"deproj": self.cus_deproj.state_dict(),}, os.path.join(path, filename))
    
    def save_soft_prompt(self, path, filename="soft_prompt.pt"):
        prompt_encoder = self.peft_model.prompt_encoder
        torch.save(prompt_encoder.state_dict(), os.path.join(path, filename))
    
    def save_pretrained(self, save_directory, **kwargs):
        self.peft_model.save_pretrained(save_directory, **kwargs)
        self.save_embeddings(save_directory)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.peft_model, name)

def freeze_model_for_emb_ft(model):
    for n, p in model.named_parameters():
        if "cus_proj" in n or "cus_deproj" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    tot_params = sum(p.numel() for p in model.parameters())
    print("***** 模型总参数数量: {} *****".format(tot_params))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("***** 模型可训练参数数量: {} *****".format(trainable_params))
    print("***** 可训练参数比例: {} % *****".format(trainable_params/tot_params * 100))
    return model

def prepare_input(data):
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to("cuda")
    return data

def spectral_norm(weight: torch.Tensor, iters: int = 20) -> float:
    W = weight.detach().to(dtype=torch.float32)
    u = torch.randn(W.size(0), device=W.device); u = u / (u.norm() + 1e-12)
    for _ in range(iters):
        v = torch.mv(W.t(), u); v = v / (v.norm() + 1e-12)
        u = torch.mv(W, v);     u = u / (u.norm() + 1e-12)
    return float(u.dot(W @ v).item())

# ====================== Prefix-LM 掩码（严格，无回退） ======================
def build_prefix_lm_mask_batch(input_ids: torch.Tensor, model) -> torch.Tensor:
    B, T = input_ids.shape
    device = input_ids.device
    mask = torch.zeros((B, T), dtype=torch.float32, device=device)

    u16 = model._unused16_id
    u17 = model._unused17_id
    u10 = model._unused10_id

    for b in range(B):
        ids = input_ids[b]

        # 找三标记起点（u16,u16,u17）
        start_pos = -1
        for t in range(T-2):
            if int(ids[t]) == u16 and int(ids[t+1]) == u16 and int(ids[t+2]) == u17:
                start_pos = t
                break
        if start_pos < 0:
            raise RuntimeError(f"样本 {b}: 未找到连续的 [unused16][unused16][unused17] 起点")

        # 从 start_pos 之后找闭合 [unused10]
        close_pos = -1
        for t in range(start_pos + 3, T):
            if int(ids[t]) == u10:
                close_pos = t
                break
        if close_pos < 0:
            raise RuntimeError(f"样本 {b}: 未找到闭合 [unused10]")

        mask[b, start_pos: close_pos + 1] = 1.0

    return mask


def prepare_input_and_label(model, input_ids):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    padded_input_tokens = model._extend_labels(input_ids)       # (B, soft_len + T)
    labels = padded_input_tokens[..., 1:].contiguous()          # (B, soft_len + T - 1)
    input_tokens = padded_input_tokens[..., :-1].contiguous()

    # 构造严格掩码（仅 R 区 + 标记）
    raw_mask = build_prefix_lm_mask_batch(input_ids, model)     # (B, T)

    # 与 soft prompt 对齐：前面补 soft_len 个 0
    if model._soft_len > 0:
        full_mask = torch.cat([torch.zeros((raw_mask.size(0), model._soft_len),
                                           dtype=raw_mask.dtype, device=raw_mask.device), raw_mask], dim=1)
    else:
        full_mask = raw_mask

    # 与 labels 对齐（右移一位）
    label_mask = full_mask[..., 1:]                             # (B, soft_len + T - 1)
    labels = labels.masked_fill(label_mask == 0, -100)
    labels[input_tokens < 0] = -100
    return labels

def loss_func(logits, inputs_ids, model, loss_fct):
    labels = prepare_input_and_label(model, inputs_ids)
    logits = logits[..., :-1, :].contiguous()
    batch_size, seq_len, vocab_size = logits.shape
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(batch_size, -1).sum(dim=-1)
    loss = loss.mean()
    return loss

@torch.no_grad()
def evaluate(prompt_model, val_loader, loss_fct, args):
    prompt_model.eval()
    nlls = []
    proj_weight=prompt_model.cus_proj.weight
    deproj_weight=prompt_model.cus_deproj.weight
    print(f"投影层权重的谱范数: {spectral_norm(proj_weight):.6f}")
    print(f"反投影层权重的谱范数: {spectral_norm(deproj_weight):.6f}")
    
    for idx, inputs_ids in tqdm(enumerate(val_loader), desc="评估中"):
        inputs_ids = inputs_ids.to("cuda")
        bs, seqlen = inputs_ids.shape
        
        labels = prepare_input_and_label(prompt_model, inputs_ids)
        output = prompt_model(input_ids=inputs_ids)
        shift_logits = output.logits[:, :-1, :]
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.shape[-1]), labels.view(-1))
        neg_log_likelihood = loss.float().reshape(bs, -1).mean(dim=-1) * seqlen
        nlls.append(neg_log_likelihood.view(1, -1))
        
    nlls = torch.hstack(nlls).view(-1)
    ppl = torch.exp(nlls.sum() / (nlls.numel() * args.seqlen))
    print(f"验证困惑度: {ppl.item():.3f}")
    return ppl.item()

def create_dp_emb_ft_model(args):
    print(f"加载基础模型: {args.model_name_or_path}")
    model_local_path = f"{PANGU_PATH}/{args.model_name_or_path}"

    # 加载 tokenizer 并确保特殊标记存在
    tokenizer = AutoTokenizer.from_pretrained(
        model_local_path, 
        use_fast=False, 
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    add_tokens = []
    for tok in ["[unused16]", "[unused17]", "[unused10]"]:
        if tok not in tokenizer.get_vocab():
            add_tokens.append(tok)
    if add_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        print(f"Added additional_special_tokens: {add_tokens}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_local_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cuda",
        local_files_only=True
    )
    print(base_model.device, base_model)
    if add_tokens:
        base_model.resize_token_embeddings(len(tokenizer))
    
    # 配置 soft prompt（若启用）
    if args.soft:
        if args.init_from_vocab:
            init_method = PromptTuningInit.TEXT
            prompt_text = "The inputs are abrupted by the noise and you should try to recover them and generate the next token."
        else:
            init_method = PromptTuningInit.RANDOM
            prompt_text = None
        
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=init_method,
            num_virtual_tokens=args.soft_token_num,
            tokenizer_name_or_path=model_local_path,
            prompt_tuning_init_text=prompt_text,
            tokenizer_kwargs={"trust_remote_code": True}
        )
        peft_model = get_peft_model(base_model, peft_config)
    else:
        peft_model = base_model
    
    # DP/Emb-FT 参数封装（原样）
    dp_args = argparse.Namespace(
        device=args.device,
        soft_token_num=args.soft_token_num,
        soft=args.soft,
        dp=args.dp,
        mu=args.mu,
        sparsity=args.sparsity,
        noise_mechanism=args.noise_mechanism,
        quant_level=args.quant_level,
        clip_bound=args.clip_bound,
        dp_rounds=args.dp_rounds,
        hard=args.hard,
        turn_to_token=args.turn_to_token,
        fixed=args.fixed,
        composition=args.composition,
        model=args.model,
        proj_dim=args.proj_dim,
        eta=args.eta,
        emb_ckpt=args.emb_ckpt,
        emb_ft=args.emb_ft,
        sp=args.sp,
        no_proj=args.no_proj
    )
    
    dp_model = DPEmbeddingFTModel(peft_model, dp_args)
    if args.emb_ft:
        dp_model = freeze_model_for_emb_ft(dp_model)
    elif args.soft:
        if args.soft_ckpt is not None:
            print(f"加载soft prompt检查点: {args.soft_ckpt}")
            prompt_encoder = peft_model.prompt_encoder
            prompt_encoder.load_state_dict(torch.load(args.soft_ckpt, map_location=torch.device("cpu")))
        for param in dp_model.parameters():
            param.requires_grad = False
        for param in peft_model.prompt_encoder.parameters():
            param.requires_grad = True
        tot_params = sum(p.numel() for p in dp_model.parameters())
        trainable_params = sum(p.numel() for p in dp_model.parameters() if p.requires_grad)
        print("***** 模型总参数数量: {} *****".format(tot_params))
        print("***** 模型可训练参数数量: {} *****".format(trainable_params))
        print("***** 可训练参数比例: {} % *****".format(trainable_params/tot_params * 100))
    
    # 注入掩码元信息
    unused16_id = tokenizer.convert_tokens_to_ids("[unused16]")
    unused17_id = tokenizer.convert_tokens_to_ids("[unused17]")
    unused10_id = tokenizer.convert_tokens_to_ids("[unused10]")
    soft_len    = args.soft_token_num if args.soft else 0
    dp_model.set_io_format_meta(unused16_id, unused17_id, unused10_id, suffix_ids=[], soft_len=soft_len)
    
    return dp_model, tokenizer

# ====================== 主入口（训练循环保持原逻辑） ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Embedding Fine-tuning with DP")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["wikitext2", "wikitext103", "ptb", "c4"])
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soft", action='store_true', default=False)
    parser.add_argument("--hard", action='store_true', default=False)
    parser.add_argument("--turn-to-token", action='store_true', default=False)
    parser.add_argument("--fixed", action='store_true', default=False)
    parser.add_argument("--composition", action='store_true', default=False)
    parser.add_argument("--max_steps", default=5000, type=int)
    parser.add_argument("--prompt_lr", type=float, default=0.3)
    parser.add_argument("--warmup_step_prompt", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--init_from_vocab", action="store_true", default=True)
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument("--soft_token_num", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="Adafactor")
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--root", type=str, default="./output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--dp", action='store_true', default=False)
    parser.add_argument("--mu", type=float, default=10000, help="DP级别")
    parser.add_argument("--sparsity", type=float, default=1.0, help="稀疏率")
    parser.add_argument("--quant_level", type=int, default=32)
    parser.add_argument("--noise_mechanism", type=str, default="Gaussian", 
                        choices=["Gaussian", "ChiDP", "Ternary", "Gaussian_binary"])
    parser.add_argument("--eta", type=float, default=500, help="epsilon-DP level")
    parser.add_argument("--clip_bound", type=float, default=1.0, help="GDP level")
    parser.add_argument("--dp-rounds", type=int, default=1, help="round of dp")
    parser.add_argument("--device", type=str, default="cuda:0",
                        choices=["cuda", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--emb_ft", action='store_true', default=False)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--emb_ckpt", type=str, default=None)
    parser.add_argument("--soft_ckpt", type=str, default=None)
    parser.add_argument("--sp", action='store_true', default=False)
    parser.add_argument("--adv", action='store_true', default=False)
    parser.add_argument("--no_proj", action='store_true', default=False)

    # —— 新增：训练时随机前缀比例区间 ——
    parser.add_argument("--prefix_ratio_min", type=float, default=0.03,
                        help="最小前缀比例（用于随机切分 P|R）")
    parser.add_argument("--prefix_ratio_max", type=float, default=0.2,
                        help="最大前缀比例（用于随机切分 P|R）")

    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = f"{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录配置
    config_str = "="*20 + "\n"
    config_str += f"dataset: {args.dataset}\n"
    config_str += f"model: {args.model}\n"
    config_str += f"seed: {args.seed}\n"
    config_str += f"init_from_vocab: {args.init_from_vocab}\n"
    config_str += f"soft_token_num: {args.soft_token_num}\n"
    config_str += f"prompt_lr: {args.prompt_lr}\n"
    config_str += f"warmup_steps: {args.warmup_step_prompt}\n"
    config_str += f"dp: {args.dp}\n"
    config_str += f"mu: {args.mu}\n"
    config_str += f"noise_mechanism: {args.noise_mechanism}\n"
    config_str += f"emb_ft: {args.emb_ft}\n"
    config_str += f"proj_dim: {args.proj_dim}\n"
    config_str += f"prefix_ratio: [{args.prefix_ratio_min}, {args.prefix_ratio_max}]\n"
    print(config_str)
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        f.write(config_str)
    
    # 创建模型和tokenizer
    model, tokenizer = create_dp_emb_ft_model(args)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    if args.dataset == "wikitext2":
        raw_train_data = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        raw_val_data = load_dataset('wikitext', 'wikitext-2-v1', split='validation')
        col_key = 'text'
        cutoff = None
    elif args.dataset == "wikitext103":
        raw_train_data = load_dataset('wikitext', 'wikitext-103-v1', split='train')
        raw_val_data = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
        col_key = 'text'
        cutoff = None
    elif args.dataset == "ptb":
        raw_train_data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        raw_val_data = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
        col_key = 'sentence'
        cutoff = None
    elif args.dataset == 'c4':
        ds = load_dataset(
            "allenai/c4", "en",
            data_files={
                "train": [
                    "en/c4-train.00000-of-01024.json.gz",
                    "en/c4-train.00001-of-01024.json.gz",
                    "en/c4-train.00002-of-01024.json.gz",
                    "en/c4-train.00004-of-01024.json.gz",
                    "en/c4-train.00005-of-01024.json.gz",
                    # "en/c4-train.00006-of-01024.json.gz",
                    # "en/c4-train.00007-of-01024.json.gz",
                    # "en/c4-train.00008-of-01024.json.gz",
                ],
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            }, 
            verification_mode="no_checks",
        )
        raw_train_data = ds["train"]
        raw_val_data = ds["validation"]
        col_key = 'text'
        cutoff = 40000  # 限制训练数据量
    
    # 数据加载器
    train_dataset = TextDataset(raw_train_data, tokenizer, args, mode="train", col_key=col_key, cutoff=cutoff)
    val_dataset = TextDataset(raw_val_data, tokenizer, args, mode="val", col_key=col_key, cutoff=1100 if args.dataset == 'c4' else None)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        shuffle=False
    )
    
    # 优化器与调度器（原逻辑）
    optimizer_group = [{
        "params": [p for n, p in model.named_parameters() if p.requires_grad],
        "weight_decay": 0.0
    }]
    
    print("训练投影和反投影参数")
    
    if args.optimizer.lower() == "adafactor":
        optimizer = Adafactor(
            optimizer_group,
            lr=args.prompt_lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=1e-5
        )
        scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=args.warmup_step_prompt
        )
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_group, 
            lr=args.prompt_lr, 
            weight_decay=1e-5
        )
        if 'pangu' in args.model.lower():
            args.warmup_step_prompt = 2000
            optimizer = torch.optim.AdamW(
                optimizer_group, 
                lr=args.prompt_lr, 
                weight_decay=1e-4, betas=(0.9, 0.95)
            )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_step_prompt,
            num_training_steps=args.max_steps
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    tot_loss = 0
    log_loss = 0
    tot_clean_loss=0
    log_clean_loss=0
    tot_adv_loss=0
    log_adv_loss=0
    tot_cos_loss=0
    log_cos_loss=0
    best_val_ppl = float('inf')
    glb_step = 0
    actual_step = 0
    leave_training = False
    tot_train_time = 0
    pbar_update_freq = 10
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    model.train()
    pbar = tqdm(total=args.max_steps, desc="训练")
    
    for epoch in range(1000000):
        print(f"开始第 {epoch} 轮训练")
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch.to("cuda")
            start_time = time.time()
            
            model.first_step = True
            
            # 前向
            model.set_adv(False)
            output = model(input_ids=input_ids)
            logits = output.logits
            loss = loss_func(logits, input_ids, model, loss_fct)

            # 可选对抗/隐私增强（原逻辑保留）
            if args.adv and args.emb_ft and glb_step>3000:
                frac = (glb_step - 3000) / (args.max_steps - 3000)
                w_clean = 1.0
                w_adv = min(frac / 0.2, 1.0) * 0.15
                w_cos = min(frac / 0.2, 1.0) * 0.8
                model.set_adv(True)
                output_adv = model(input_ids=input_ids)
                logits_adv = output_adv.logits
                loss_adv = loss_func(logits_adv, input_ids, model, loss_fct)
                loss_cos = model.cos_sim
                tot_clean_loss += loss.item()
                loss = 1.0*loss + w_cos*loss_cos
                tot_adv_loss += loss_adv.item()
                tot_cos_loss += loss_cos.item()
            
            # 反向与更新
            loss.backward()
            tot_loss += loss.item()
            actual_step += 1
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 
                1.0
            )
            glb_step += 1
            
            if glb_step % pbar_update_freq == 0:
                avg_loss = (tot_loss - log_loss) / pbar_update_freq
                avg_clean_loss = (tot_clean_loss - log_clean_loss) / pbar_update_freq
                avg_adv_loss = (tot_adv_loss - log_adv_loss) / pbar_update_freq
                avg_cos_loss = (tot_cos_loss - log_cos_loss) / pbar_update_freq
                pbar.update(pbar_update_freq)
                if args.adv and args.emb_ft and glb_step>3000:
                    pbar.set_postfix({'总损失': avg_loss, 'clean': avg_clean_loss, 'adv': avg_adv_loss, 'cos': avg_cos_loss})
                    log_clean_loss = tot_clean_loss
                    log_adv_loss = tot_adv_loss
                    log_cos_loss = tot_cos_loss
                else:
                    pbar.set_postfix({'loss': avg_loss})
                log_loss = tot_loss
                
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            tot_train_time += time.time() - start_time
            
            # 评估
            if glb_step > 0 and glb_step % args.eval_every_steps == 0:
                val_ppl = evaluate(model, val_dataloader, loss_fct, args)
                print(f"步骤 {glb_step}: val_ppl = {val_ppl:.3f}")
                
                if val_ppl <= best_val_ppl:
                    if args.emb_ft and not args.sp:
                        model.save_embeddings(path=output_dir, filename=f"projs.pth")
                    elif args.sp:
                        proj_state_dict = model.cus_proj.state_dict()
                        deproj_state_dict = model.cus_deproj.state_dict()
                        baked_proj_state_dict = bake_spectral_norm_weights(proj_state_dict)
                        baked_deproj_state_dict = bake_spectral_norm_weights(deproj_state_dict)
                        torch.save({"proj": baked_proj_state_dict,"deproj": baked_deproj_state_dict,}, f"{output_dir}/projs.pth")
                    else:
                        model.save_soft_prompt(path=output_dir, filename=f"mu_{args.mu}_best.pt")
                        print(f"保存最佳soft prompt到 {output_dir}/mu_{args.mu}_best.pt")
                    best_val_ppl = val_ppl
                    print(f"新的最佳验证困惑度: {best_val_ppl:.3f}")
                
                print(f"步骤 {glb_step}, val_ppl {val_ppl:.3f}, 平均时间 {tot_train_time/actual_step:.4f}秒/步")
                model.train()
            
            if glb_step > args.max_steps:
                leave_training = True
                break
        
        if leave_training:
            break
    
    final_val_ppl = evaluate(model, val_dataloader, loss_fct, args)
    print(f"最终验证困惑度: {final_val_ppl:.3f}")
    print(f"最佳验证困惑度: {best_val_ppl:.3f}")

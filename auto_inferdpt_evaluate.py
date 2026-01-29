#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import json
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

PANGU_PATH = os.getenv("PANGU_PATH", "/default/pangu/path")

def safe_tag(model_name_or_path: str) -> str:
    tag = model_name_or_path.strip().replace("\\", "/")
    for ch in ["/", ":", " ", "@", "#", "?", "&", "=", "+"]:
        tag = tag.replace(ch, "__")
    return tag

def load_prompts_for_search(dataset: str,
                 max_samples: Optional[int] = None,
                 file_path: Optional[str] = None) -> List[str]:
    """
    支持:
      - 'ptb'        : HF "ptb_text_only" 测试集 sentence 字段
      - 'wikitext2'  : HF "wikitext-2-v1" 测试集 text 字段（去空）
      - 'file'       : 本地 .txt 或 .jsonl（每行一个JSON，缺省字段 'text'）
    """
    ds = dataset.lower()
    texts: List[str] = []

    if ds == "ptb":
        from datasets import load_dataset
        try:
            raw = load_dataset("ptb_text_only", "penn_treebank", split="test")
            field = "sentence"
        except Exception:
            raw = load_dataset("ptb_text_only", split="test")
            field = "sentence"
        for item in raw:
            s = str(item.get(field, "")).strip()
            if s:
                texts.append(s)

    elif ds == "wikitext2":
        from datasets import load_dataset
        raw = load_dataset("wikitext", "wikitext-2-v1", split="test")
        for s in raw["text"]:
            s = str(s).strip()
            if s:
                texts.append(s)

    elif ds == "file":
        if not file_path:
            raise ValueError("--file_path is required when dataset='file'")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        if file_path.endswith(".jsonl"):
            text_key = os.environ.get("FILE_TEXT_KEY", "text")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        s = str(obj.get(text_key, "")).strip()
                        if s:
                            texts.append(s)
                    except Exception:
                        continue
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        texts.append(s)
        else:
            raise ValueError("Only .jsonl or .txt supported for dataset='file'")
    else:
        raise ValueError("dataset must be one of {'ptb','wikitext2','file'}")

    if max_samples is not None and len(texts) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in idx]

    print(f"[data] loaded {len(texts)} prompts from '{dataset}'")
    return texts


def load_prompts(dataset: str,
                 max_samples: Optional[int] = None,
                 file_path: Optional[str] = None) -> List[str]:
    ds = dataset.lower()
    texts: List[str] = []

    if ds == "ptb":
        from datasets import load_dataset
        try:
            raw = load_dataset("ptb_text_only", "penn_treebank", split="test")
            field = "sentence"
        except Exception:
            raw = load_dataset("ptb_text_only", split="test")
            field = "sentence"
        for item in raw:
            s = str(item.get(field, "")).strip()
            if s:
                texts.append(s)

    elif ds == "wikitext2":
        from datasets import load_dataset
        raw = load_dataset("wikitext", "wikitext-2-v1", split="test")
        for s in raw["text"]:
            s = str(s).strip()
            if s:
                texts.append(s)

    elif ds == "file":
        if not file_path:
            raise ValueError("--file_path is required when dataset='file'")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        if file_path.endswith(".jsonl"):
            text_key = os.environ.get("FILE_TEXT_KEY", "text")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        s = str(obj.get(text_key, "")).strip()
                        if s:
                            texts.append(s)
                    except Exception:
                        continue
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        texts.append(s)
        else:
            raise ValueError("Only .jsonl or .txt supported for dataset='file'")
    else:
        raise ValueError("dataset must be one of {'ptb','wikitext2','file'}")

    if max_samples is not None and len(texts) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), size=max_samples, replace=False)
        texts = [texts[i] for i in idx]

    print(f"[data] loaded {len(texts)} prompts from '{dataset}'")
    return texts

def _get_model_ctx_len(model, tokenizer) -> int:
    # 可能的字段：max_position_embeddings / n_positions / tokenizer.model_max_length
    ctx = None
    for k in ("max_position_embeddings", "n_positions"):
        if hasattr(model.config, k) and getattr(model.config, k):
            ctx = int(getattr(model.config, k))
            break
    # tokenizer 的值有时是非常大(1e12)的哨兵，需过滤
    tmax = getattr(tokenizer, "model_max_length", None)
    if tmax and isinstance(tmax, int) and tmax < 100_000:
        ctx = min(ctx, tmax) if ctx else tmax
    if not ctx:
        ctx = 2048  # 合理保守默认
    return int(ctx)

def _ensure_pad_token(tokenizer):
    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos

def _clip_text_tokens(tokenizer, text: str, max_tokens: int, keep: str = "right") -> str:
    """
    将 text 截断到不超过 max_tokens 个token。
    keep="right" 保留结尾（适合 decoder-only），"left" 保留开头。
    """
    if max_tokens <= 0:
        return ""
    old_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left" if keep == "right" else "right"
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    tokenizer.truncation_side = old_side
    return tokenizer.decode(ids, skip_special_tokens=False)


# ============================================================
# 嵌入缓存加载（.npy/.json；来自 HF 模型导出）
# ============================================================
def load_embedding_cache(
    data_dir: str,
    emb_npy: str,
    tok_json: str,
    sen_npy: str,
    dtype: str = "float32",
):

    emb_path = os.path.join(data_dir, emb_npy)
    tok_path = os.path.join(data_dir, tok_json)
    sen_path = os.path.join(data_dir, sen_npy)

    if not (os.path.exists(emb_path) and os.path.exists(tok_path) and os.path.exists(sen_path)):
        raise FileNotFoundError(
            f"Missing cache files under {data_dir}. "
            f"Expected: {emb_npy}, {tok_json}, {sen_npy}. "
            f"Use the export script to create them from a HF model."
        )

    want = np.float32 if dtype == "float32" else np.float64
    emb = np.load(emb_path, mmap_mode="r")
    if emb.dtype != want:
        emb = emb.astype(want, copy=False)

    with open(tok_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)                         # list[str]，id -> token
    delta = np.load(sen_path)                         # (d,) float64
    if dtype == "float64":
        delta = delta.astype(np.float64, copy=False)
    else:
        delta = delta.astype(np.float64, copy=False)

    emb_norm2 = np.einsum("ij,ij->i", emb, emb)       # (V,)

    return emb, tokens, delta, emb_norm2


# ============================================================
# 资源封装（HF 分词器 + GPU 批量 GEMM；id == 行号）
# ============================================================
class InferDPT_HFResources:
    def __init__(self,
                 hf_tokenizer_name: str,
                 emb_matrix: np.ndarray,
                 tokens: list[str],
                 delta_vec: np.ndarray,
                 emb_norm2: np.ndarray,
                 device: Optional[str] = None,
                 dtype: str = "float32"):
        # if 'llama' in hf_tokenizer_name.lower():
        #     self.tok = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=True)
        # else:
        if 'pangu' in hf_tokenizer_name.lower():
            model_local_path = f"{PANGU_PATH}/{hf_tokenizer_name}"


            # load the tokenizer and the model
            self.tok = AutoTokenizer.from_pretrained(
                model_local_path, 
                use_fast=False, 
                trust_remote_code=True,
                local_files_only=True
            )

        else:
            self.tok = AutoTokenizer.from_pretrained(hf_tokenizer_name, use_fast=True)
        if hasattr(self.tok, "pad_token") and self.tok.pad_token is None:
            try:
                self.tok.pad_token = self.tok.eos_token
            except Exception:
                pass

        if hasattr(torch, "npu") and torch.npu.is_available():
            # 昇腾 NPU
            self.device = torch.device("npu")
        elif torch.cuda.is_available():
            # 如果将来这个代码换到有 CUDA 的机器上，也能直接用
            self.device = torch.device("cuda")
        else:
            # 没 NPU 也没 CUDA，就用 CPU
            self.device = torch.device("cpu")
        # self.device = torch.device(device)

        want_torch = torch.float32 if dtype == "float32" else torch.float64
        want_np = np.float32 if dtype == "float32" else np.float64

        # 关键：做一个新的、可写、连续的 array，避免只读 memmap
        emb_np = np.array(emb_matrix, dtype=want_np, copy=True)  # ★ 必须 copy=True

        # 如果显存压力大，可以在这里换成 float16
        # emb_np = emb_np.astype(np.float16, copy=False)
        # want_torch = torch.float16

        # 同步拷贝到设备，不要 non_blocking
        self.emb_t = torch.from_numpy(emb_np).to(self.device)  # [V, d]

        # emb_norm2 的 dtype 也对齐过来
        self.emb_norm2_t = torch.from_numpy(
            emb_norm2.astype(emb_np.dtype, copy=False)
        ).to(self.device)  # [V]

        self.delta_vec = delta_vec.astype(np.float64, copy=False)  # CPU double
        self.tokens = tokens
        self.V, self.d = emb_np.shape
        self.torch_dtype = want_torch

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        enc = self.tok(texts, add_special_tokens=False,
                       return_attention_mask=False, return_token_type_ids=False)
        return [list(map(int, ids)) for ids in enc["input_ids"]]

    @torch.no_grad()
    def all_dists_from_origins_batch(self, origin_tids: torch.Tensor) -> torch.Tensor:
        E_O = self.emb_t.index_select(0, origin_tids)              # [B, d]
        v_norm2 = (E_O * E_O).sum(dim=1)                           # [B]
        dots = self.emb_t @ E_O.t()                                # [V, B]
        dist2 = self.emb_norm2_t.unsqueeze(1) + v_norm2.unsqueeze(0) - 2.0 * dots
        dist2.clamp_(min=0.0)
        return torch.sqrt(dist2)                                   # [V, B]

    @torch.no_grad()
    def sample_noise_norms_exact(self, n_tokens: int, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        tt = 0.0
        if (epsilon * 19.064721649556482 - 38.1294334077209) > 0:
            tt = 0.01658160142016071 * np.log(epsilon * 19.064721649556482 - 38.1294334077209) + 9.311083811697406
        beta = (self.delta_vec / epsilon) if epsilon < 2 else (self.delta_vec / tt)  # (d,)
        noise = rng.laplace(loc=0.0, scale=beta, size=(n_tokens, self.d))
        norms = np.sqrt((noise * noise).sum(axis=1))
        return norms


# ============================================================
# 批量扰动（GPU批处理）
# ============================================================
def perturb_sentences_batch_fast_with_counts(
    sentences: List[str],
    epsilon: float,
    resources: InferDPT_HFResources,
    rng: Optional[np.random.Generator] = None,
    origin_batch_size: int = 128,
    numeric_strategy: str = "random",
) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    if rng is None:
        rng = np.random.default_rng(12345)

    token_id_lists = resources.encode_batch(sentences)

    token_ids_flat = np.concatenate([np.asarray(tids, dtype=np.int64) for tids in token_id_lists], axis=0)
    n_tokens = token_ids_flat.shape[0]
    if n_tokens == 0:
        return sentences, [0]*len(sentences), [0]*len(sentences), [0]*len(sentences), [0]*len(sentences)

    out_tokens_flat = np.array([resources.tokens[int(tid)] for tid in token_ids_flat], dtype=object)
    is_numeric_flat = np.array([resources.tokens[int(tid)].strip().isdigit() for tid in token_ids_flat], dtype=bool)
    norms = resources.sample_noise_norms_exact(n_tokens, epsilon, rng)

    Delta_u = 1.0
    exp_factor = epsilon / (2.0 * Delta_u)

    changed_flags = np.zeros(n_tokens, dtype=bool)

    positions_by_origin: Dict[int, List[int]] = {}
    has_embed = np.ones(n_tokens, dtype=bool)
    for idx, tid in enumerate(token_ids_flat.tolist()):
        if tid < 0 or tid >= resources.V:
            has_embed[idx] = False
            continue
        positions_by_origin.setdefault(int(tid), []).append(idx)

    coverage_total = int(has_embed.sum())
    coverage_ratio = coverage_total / n_tokens if n_tokens > 0 else 0.0
    print(f"[stats] tokens_total={n_tokens}, can_perturb={coverage_total} ({coverage_ratio:.2%})")

    origin_ids = np.array(list(positions_by_origin.keys()), dtype=np.int64)
    device = resources.device

    for b in range(0, origin_ids.size, origin_batch_size):
        batch_o = origin_ids[b:b+origin_batch_size]
        batch_o_t = torch.from_numpy(batch_o).to(device)

        dists_VB = resources.all_dists_from_origins_batch(batch_o_t)   # [V, B]
        dists_VB_cpu = dists_VB.detach().cpu().numpy()                 # (V, B)

        for col, origin_tid in enumerate(batch_o.tolist()):
            pos_list = positions_by_origin[origin_tid]
            dcol = dists_VB_cpu[:, col]                                 # (V,)
            cand_all_idx = np.arange(resources.V, dtype=np.int32)

            for p in pos_list:
                r = float(norms[p])
                if r <= 0.0:
                    continue

                mask = dcol <= (r + 1e-12)
                if not np.any(mask):
                    continue

                cand_idx = cand_all_idx[mask]
                cand_d   = dcol[mask].astype(np.float64)

                alpha = exp_factor / r
                w = np.exp(-alpha * cand_d)
                s = w.sum()
                if s <= 0 or not np.isfinite(s):
                    continue
                w /= s

                cdf = np.cumsum(w)
                u = float(np.random.random())
                choice_pos = int(np.searchsorted(cdf, u, side="right"))
                if choice_pos >= cand_idx.size:
                    choice_pos = cand_idx.size - 1

                chosen_idx = int(cand_idx[choice_pos])
                chosen_tok = resources.tokens[chosen_idx]
                if chosen_tok != out_tokens_flat[p]:
                    out_tokens_flat[p] = chosen_tok
                    changed_flags[p] = True

    if numeric_strategy == "random" and is_numeric_flat.any():
        rand_nums = np.random.randint(low=1, high=1001, size=int(is_numeric_flat.sum()))
        out_tokens_flat[is_numeric_flat] = rand_nums.astype(str)
        changed_flags[is_numeric_flat] = True

    outputs: List[str] = []
    per_total: List[int] = []
    per_unperturbed: List[int] = []
    per_cover: List[int] = []
    per_changed_on_cover: List[int] = []

    ptr = 0
    for tids in token_id_lists:
        L = len(tids)
        toks = out_tokens_flat[ptr:ptr+L].tolist()
        flags = changed_flags[ptr:ptr+L]
        cover_mask = has_embed[ptr:ptr+L]
        changed_mask = changed_flags[ptr:ptr+L]

        outputs.append(' '.join(toks))
        per_total.append(int(L))
        per_unperturbed.append(int((~flags).sum()))
        per_cover.append(int(cover_mask.sum()))
        per_changed_on_cover.append(int((changed_mask & cover_mask).sum()))
        ptr += L

    return outputs, per_total, per_unperturbed, per_cover, per_changed_on_cover


# ============================================================
# 二分搜索 ε
# ============================================================
def binary_search_epsilon_for_target_attack_rate(
    dataset_prompts: List[str],
    target_attack_rate: float,
    resources: InferDPT_HFResources,
    eps_low: float = 0.1,
    eps_high: float = 30.0,
    tolerance: float = 0.01,
    max_iterations: int = 15,
    sample_size: Optional[int] = 2000,
    seed: int = 42,
    origin_batch_size: int = 128,
    numeric_strategy: str = "random",
) -> Tuple[float, float]:
    assert 0.0 <= target_attack_rate <= 1.0

    eval_prompts = list(dataset_prompts)
    if sample_size is not None and len(eval_prompts) > sample_size:
        rng_sel = np.random.default_rng(seed)
        idx = rng_sel.choice(len(eval_prompts), size=sample_size, replace=False)
        eval_prompts = [eval_prompts[i] for i in idx]

    best_eps, best_rate, best_diff = None, None, float("inf")

    print("\n=== InferDPT ε-search (online, exact, GPU batch) ===")
    print(f"Target attack success rate: {target_attack_rate:.3f}")
    print(f"Search range: ε ∈ [{eps_low}, {eps_high}]  (tolerance={tolerance})")
    print("-" * 60)

    low, high = eps_low, eps_high
    for it in range(1, max_iterations + 1):
        mid = 0.5 * (low + high)

        _, per_total, _, per_cover, per_changed_on_cover = perturb_sentences_batch_fast_with_counts(
            sentences=eval_prompts,
            epsilon=mid,
            resources=resources,
            origin_batch_size=origin_batch_size,
            numeric_strategy=numeric_strategy,
        )

        total_cover = int(np.sum(per_cover))
        total_changed_on_cover = int(np.sum(per_changed_on_cover))
        cur_rate = (total_cover - total_changed_on_cover) / total_cover if total_cover > 0 else 0.0

        print(f"[Iter {it:02d}] ε={mid:7.4f} | attack_rate={cur_rate:.4f} | cover={total_cover}/{np.sum(per_total)} "
              f"({total_cover/np.sum(per_total):.2%})")

        diff = abs(cur_rate - target_attack_rate)
        if diff < best_diff:
            best_diff = diff
            best_eps = mid
            best_rate = cur_rate

        if diff <= tolerance:
            print(f"✓ reached tolerance at iteration {it}")
            break

        if cur_rate < target_attack_rate:
            low = mid
        else:
            high = mid

    print("-" * 60)
    print(f"Done. Best ε={best_eps:.4f}, attack_rate={best_rate:.4f}, diff={best_diff:.4f}")
    return best_eps, best_rate


def auto_search_epsilon_inferdpt_online(
    dataset_prompts: List[str],
    target_attack_rates: List[float],
    resources: InferDPT_HFResources,
    eps_low: float = 0.1,
    eps_high: float = 30.0,
    tolerance: float = 0.01,
    max_iterations: int = 15,
    sample_size: Optional[int] = 2000,
    origin_batch_size: int = 128,
    numeric_strategy: str = "random",
) -> Dict[float, Dict[str, float]]:
    results: Dict[float, Dict[str, float]] = {}
    for tar in target_attack_rates:
        beps, brate = binary_search_epsilon_for_target_attack_rate(
            dataset_prompts=dataset_prompts,
            target_attack_rate=tar,
            resources=resources,
            eps_low=eps_low,
            eps_high=eps_high,
            tolerance=tolerance,
            max_iterations=max_iterations,
            sample_size=sample_size,
            origin_batch_size=origin_batch_size,
            numeric_strategy=numeric_strategy,
        )
        results[tar] = {"best_eps": float(beps), "attack_rate": float(brate)}
    return results


def export_results_to_csv(results: Dict[float, Dict[str, float]], path: str, gptq_tags: List[str]) -> None:
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        # 基础列
        header = ["Target_Attack_Rate", "Best_Epsilon", "Actual_Attack_Rate", "SimCSE_Before"]
        # 为每个 GPTQ 模型添加一列：SimCSE_After__{tag}
        for tag in gptq_tags:
            header.append(f"SimCSE_After__{tag}")
        w.writerow(header)

        for tar, obj in results.items():
            row = [tar, obj["best_eps"], obj["attack_rate"], obj.get("simcse_before", "")]
            for tag in gptq_tags:
                row.append(obj.get(f"simcse_after__{tag}", ""))
            w.writerow(row)
    print(f"[export] results saved to: {path}")



def preprocess_dataset_for_eval(dataset_name: str, max_samples: int = 2000,
                                file_path: Optional[str] = None,
                                prompt_len: int = 50, min_len: int = 100):
    """
    产出：
      - prompts: 前 prompt_len 个词
      - references: 剩余部分作为参考
    与附件脚本保持一致策略（wikitext2/PTB），file 模式做一个通用切分。
    """
    from datasets import load_dataset
    processed = []

    if dataset_name.lower() == "wikitext2":
        ds = load_dataset('wikitext', 'wikitext-2-v1', split='test')
        for item in ds:
            text = str(item['text']).strip()
            if len(text.split()) >= min_len:
                toks = text.split()
                processed.append({
                    "text": text,
                    "prompt": ' '.join(toks[:prompt_len]),
                    "reference": ' '.join(toks[prompt_len:])
                })
                if len(processed) >= max_samples:
                    break

    elif dataset_name.lower() == "ptb":
        try:
            ds = load_dataset('ptb_text_only', 'penn_treebank', split='test')
            field = 'sentence'
        except Exception:
            ds = load_dataset('ptb_text_only', split='test')
            field = 'sentence'
        for item in ds:
            text = str(item[field]).strip()
            if len(text.split()) >= min_len:
                toks = text.split()
                processed.append({
                    "text": text,
                    "prompt": ' '.join(toks[:prompt_len]),
                    "reference": ' '.join(toks[prompt_len:])
                })
                if len(processed) >= max_samples:
                    break

    elif dataset_name.lower() == "file":
        if not file_path:
            raise ValueError("--file_path is required when dataset='file'")
        raw = []
        if file_path.endswith(".jsonl"):
            key = os.environ.get("FILE_TEXT_KEY", "text")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        s = str(obj.get(key, "")).strip()
                        if s:
                            raw.append(s)
                    except Exception:
                        continue
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        raw.append(s)
        else:
            raise ValueError("Only .jsonl or .txt supported for dataset='file'")

        for text in raw:
            toks = text.split()
            if len(toks) < max(prompt_len + 1, min_len):
                continue
            processed.append({
                "text": text,
                "prompt": ' '.join(toks[:prompt_len]),
                "reference": ' '.join(toks[prompt_len:])
            })
            if len(processed) >= max_samples:
                break
    else:
        raise ValueError("dataset must be one of {'ptb','wikitext2','file'}")

    print(f"[data-eval] prepared {len(processed)} samples with prompt/reference for '{dataset_name}'")
    return processed


def load_simcse(model_name="princeton-nlp/sup-simcse-bert-base-uncased", device: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    if device is not None and str(device).lower() not in ["none", "auto", ""]:
        dev = torch.device(device)
    else:
        if hasattr(torch, "npu") and torch.npu.is_available():
            dev = torch.device("npu")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

    mdl = mdl.to(dev)
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def simcse_batch_cosine(texts1: List[str], texts2: List[str], mdl: AutoModel, tok: AutoTokenizer,
                        batch_size: int = 8) -> List[float]:
    assert len(texts1) == len(texts2)
    device = next(mdl.parameters()).device
    sims: List[float] = []
    for i in tqdm(range(0, len(texts1), batch_size), desc="Computing SimCSE"):
        b1 = texts1[i:i+batch_size]
        b2 = texts2[i:i+batch_size]
        inp1 = tok(b1, padding=True, truncation=True, return_tensors="pt").to(device)
        inp2 = tok(b2, padding=True, truncation=True, return_tensors="pt").to(device)
        out1 = mdl(**inp1).last_hidden_state[:, 0]   # [CLS]
        out2 = mdl(**inp2).last_hidden_state[:, 0]
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        bsim = torch.bmm(out1.unsqueeze(1), out2.unsqueeze(2)).squeeze().float().cpu().tolist()
        sims.extend(bsim)
    return sims


@torch.no_grad()
def generate_with_causallm(model, tokenizer, prompts: List[str], max_new_tokens: int = 200) -> List[str]:
    outs: List[str] = []
    for i, p in enumerate(tqdm(prompts, desc="Generating")):
        if 'pangu' in tokenizer.name_or_path.lower():
            p = "Continue the text."+p+' \no_think'
        enc = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
            use_cache=True,
        )

        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        # if 'pangu' in tokenizer.name_or_path.lower():
        #     gen = gen.split("[unused17]")[-1].split("[unused10]")[0].strip()
        outs.append(gen)
        # del enc, out
        # torch.cuda.empty_cache()
        if i < 5:
            print(f"\n[GEN {i+1}] prompt: {p}\n[GEN] {gen}\n")
    return outs


@torch.no_grad()
def denoise_with_gptq(gptq:bool, gptq_model_name: str,
                      texts_in: List[Tuple[str, str]],  # (prompt, generated)
                      max_new_tokens: int = 200) -> Tuple[AutoModelForCausalLM, AutoTokenizer, List[str]]:
    if not 'bit' in gptq_model_name and not 'bnb' in gptq_model_name:
        if 'llama' in gptq_model_name.lower():

            tok = AutoTokenizer.from_pretrained(gptq_model_name, use_fast=False)
            mdl = AutoModelForCausalLM.from_pretrained(gptq_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

        else:
            tok = AutoTokenizer.from_pretrained(gptq_model_name, use_fast=False)

            mdl = AutoModelForCausalLM.from_pretrained(
                gptq_model_name,
                trust_remote_code=True,
                device_map="auto",
            )
    else:
        tok = AutoTokenizer.from_pretrained(gptq_model_name, use_fast=False)

        # 手动构造量化配置：4bit + groupsize=128 + act-order + true-sequential
        qcfg = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=True,          # 等价于 act-order
            true_sequential=True,   # 仓库卡片里写了 true-sequential
            sym=True                # 常用；不设置也行
        )

        mdl = AutoGPTQForCausalLM.from_quantized(
            gptq_model_name,
            model_basename="llama-7b-4bit-128g",  # 必须与 .safetensors 前缀一致
            quantize_config=qcfg,                 # 关键：绕过缺失文件
            device_map="auto",
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_attention=False          # 某些环境关掉更稳
        )
    mdl.eval()

    ctx_len = _get_model_ctx_len(mdl, tok)
    # 留出生成空间与少量余量
    max_prompt_len = max(16, ctx_len - max_new_tokens - 8)

    # 固定的指令头与尾（尽量不被截断）
    head = (
        "Your task is to extend the \"Prefix Text\". Use the \"Perturbed Generation\" as your primary "
        "writing material for your extension. Extract coherent and consistent text from the \"Perturbed "
        "Generation\" and integrate them into your continuation. Ensure a seamless alignment with the "
        "context established by the \"Prefix Text\". Provide only your \"Extended Text\".\n"
        "——\"Prefix Text\": "
    )
    mid  = "\n——\"Perturbed Generation\": "
    tail = "\n——\"Extended Text\": "

    # 先计算头+尾+短前缀的开销；再将“Perturbed Generation”按 token 预算截断（保尾）
    head_ids = tok(head, add_special_tokens=False)["input_ids"]
    mid_ids  = tok(mid,  add_special_tokens=False)["input_ids"]
    tail_ids = tok(tail, add_special_tokens=False)["input_ids"]
    fixed_cost = len(head_ids) + len(mid_ids) + len(tail_ids)

    outs: List[str] = []
    for i, (pref, gen) in enumerate(tqdm(texts_in, desc="Denoising with GPTQ")):
        # prefix 也限制一个上限，避免极端长
        # 给 prefix 分配最多 1/2 预算（经验性分配），剩余给 gen
        budget = max_prompt_len
        pref_budget = max(8, min(len(tok(pref, add_special_tokens=False)["input_ids"]),
                               (budget - fixed_cost) // 2))
        pref_safe = _clip_text_tokens(tok, pref, max_tokens=pref_budget, keep="right")

        # 重新计算剩余预算给 gen
        used = len(tok(pref_safe, add_special_tokens=False)["input_ids"]) + fixed_cost
        gen_budget = max(8, budget - used)
        gen_safe = _clip_text_tokens(tok, gen, max_tokens=gen_budget, keep="right")

        prompt_text = f"{head}{pref_safe}{mid}{gen_safe}{tail}"
        enc = tok(prompt_text, return_tensors="pt").to(mdl.device)

        # 保险起见：再守一次上限（极小概率溢出）
        if enc["input_ids"].shape[1] > max_prompt_len:
            # 整体保尾截断到 budget
            prompt_text = _clip_text_tokens(tok, prompt_text, max_tokens=max_prompt_len, keep="right")
            enc = tok(prompt_text, return_tensors="pt").to(mdl.device)

        out = mdl.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
            use_cache=True
        )
        de = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(de)
        if i < 3:
            print(f"\n[DENOISE {i+1}] ctx={ctx_len}  prompt_tok≤{max_prompt_len}\n{de}\n")

    return mdl, tok, outs



def summarize_simcse(name: str, sims: List[float]) -> Dict[str, float]:
    arr = np.asarray(sims, dtype=np.float64) if len(sims) > 0 else np.array([0.0])
    res = {
        "avg_similarity": float(np.mean(arr)),
        "median_similarity": float(np.median(arr)),
        "min_similarity": float(np.min(arr)),
        "max_similarity": float(np.max(arr)),
    }
    print(f"\n----- SimCSE Summary ({name}) -----")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")
    return res

def save_denoised_jsonl(out_path: str, prompts: List[str], gens_before: List[str],
                        gens_after: List[str], refs: List[str], target_rate: float,
                        epsilon: float, gptq_model: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(prompts)):
            rec = {
                "gptq_model": gptq_model,
                "target_attack_rate": float(target_rate),
                "epsilon": float(epsilon),
                "prompt": prompts[i],
                "reference": refs[i],
                "gen_before": gens_before[i],
                "gen_after": gens_after[i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[export] denoised samples -> {out_path}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="InferDPT (online exact, HF + GPU batch): epsilon search + generation + SimCSE + GPTQ denoise"
    )

    # 数据集
    p.add_argument("--dataset", type=str, default="ptb", choices=["ptb", "wikitext2", "file"])
    p.add_argument("--file_path", type=str, default=None, help="path to .jsonl or .txt when dataset='file'")
    p.add_argument("--max_samples", type=int, default=200, help="max samples for eval (None for all)")

    # 嵌入缓存目录（由导出脚本生成）
    p.add_argument("--inferdpt_data_dir", type=str, default="./inferdpt/")

    # 目标模型的 HF 分词器名（需与导出用的模型一致；用于扰动词表/编码）
    p.add_argument("--hf_tokenizer", type=str, default="baffo32/decapoda-research-llama-7b-hf",
                   help="e.g., meta-llama/Llama-3-8B, Qwen/Qwen2.5-7B, facebook/opt-6.7b, gpt2")
    p.add_argument("--model_tag", type=str, default=None,
                   help="文件前缀；若不提供，将由 --hf_tokenizer 自动推导")

    # 评估/生成模型
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="用于生成的 CausalLM（before 去噪）")
    p.add_argument("--gptq_model", type=str, nargs="+", required=True,
                help="一个或多个用于去噪的 GPTQ 模型；多模型将逐一去噪与评估")

    p.add_argument("--sce_model_name", type=str, default="princeton-nlp/sup-simcse-bert-base-uncased",
                   help="SimCSE 编码器")

    # 加速与精度
    p.add_argument("--device", type=str, default='None', help="cuda / cpu (default: cuda if available)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"], help="compute/storage dtype")
    p.add_argument("--origin_batch_size", type=int, default=128, help="number of unique origins per GEMM batch")
    p.add_argument("--max_new_tokens", type=int, default=200, help="CausalLM 生成新 token 数")
    p.add_argument("--simcse_batch", type=int, default=8, help="SimCSE 计算 batch 大小")

    # 数字替换策略
    p.add_argument("--numeric_strategy", type=str, default="random", choices=["random", "keep"],
                   help="how to handle pure-numeric tokens")

    # 搜索目标
    p.add_argument("--targets", type=float, nargs="+", default=[0.02,0.1, 0.2, 0.4, 0.6],
                   help="target attack success rates")

    # 搜索范围
    p.add_argument("--eps_low", type=float, default=0.0001)
    p.add_argument("--eps_high", type=float, default=50.0)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--max_iterations", type=int, default=15)
    p.add_argument("--sample_size", type=int, default=2000, help="用于 ε 搜索与评估的样本数上限（同一子集）")
    p.add_argument("--sample_size_for_search", type=int, default=2000, help="用于 ε 搜索与评估的样本数上限（同一子集）")
    # 导出
    p.add_argument("--gptq", action="store_true", default=False, help="whether to do GPTQ denoising & eval")

    p.add_argument("--csv", type=str, default=None, help="path to save CSV (optional)")
    p.add_argument("--detail_json", type=str, default=None, help="保存每条样本的 before/after 相似度细节（可选）")
    p.add_argument("--mc", type=str, default="0", help="给输出目录添加后缀（例如：--mc 42 会生成 results_mc42）")


    return p.parse_args()


def main():
    args = parse_args()
    args.max_new_tokens = 50 if args.dataset == 'ptb' else 200
    mc_suffix   = f"_mc{args.mc}" if args.mc else ""
    results_root = f"./results{mc_suffix}"

    gptq_models: List[str] = args.gptq_model
    gptq_tags = [safe_tag(m) for m in gptq_models]


    # ========== 1) 准备“带 reference”的评估数据 ==========
    prompts_for_search = load_prompts(args.dataset, max_samples=args.sample_size_for_search, file_path=args.file_path)
    if args.dataset == 'ptb':
        processed = preprocess_dataset_for_eval(args.dataset, max_samples=args.max_samples,
                                        file_path=args.file_path, prompt_len=20, min_len=30)
    else:
        processed = preprocess_dataset_for_eval(args.dataset, max_samples=args.max_samples,
                                                file_path=args.file_path, prompt_len=50, min_len=100)
    if len(processed) == 0:
        raise RuntimeError("No samples prepared for evaluation (prompt/reference).")
    prompts_eval = [x["prompt"] for x in processed]
    refs_eval    = [x["reference"] for x in processed]

    # ========== 2) 资源（HF 模型导出的 .npy/.json）==========
    tag = args.model_tag or safe_tag(args.hf_tokenizer)
    emb_name = f"{tag}.embeddings.npy"
    tok_name = f"{tag}.tokens.json"
    sen_name = f"{tag}.sensitivity.npy"

    emb, tokens, delta_vec, emb_norm2 = load_embedding_cache(
        args.inferdpt_data_dir,
        emb_npy=emb_name,
        tok_json=tok_name,
        sen_npy=sen_name,
        dtype=args.dtype
    )
    print(f"[files] using tag={tag}")
    print(f"[files] embeddings={emb_name}  tokens={tok_name}  sensitivity={sen_name}")

    resources = InferDPT_HFResources(
        hf_tokenizer_name=args.hf_tokenizer,
        emb_matrix=emb,
        tokens=tokens,
        delta_vec=delta_vec,
        emb_norm2=emb_norm2,
        device=args.device,
        dtype=args.dtype
    )

    print(f"[sanity] |V|={resources.V}, dim={resources.d}, tokenizer={args.hf_tokenizer}, device={resources.device}")

    # ========== 3) ε 搜索（在与评估相同子集上进行）==========
    search_results = auto_search_epsilon_inferdpt_online(
        dataset_prompts=prompts_for_search,
        target_attack_rates=args.targets,
        resources=resources,
        eps_low=args.eps_low,
        eps_high=args.eps_high,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        sample_size=args.sample_size,
        origin_batch_size=args.origin_batch_size,
        numeric_strategy=args.numeric_strategy,
    )
    # ========== 4) 准备生成模型 & SimCSE ==========
    print(f"[gen] loading causal LM: {args.model_name_or_path}")
    # gen_tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if not 'bit' in args.model_name_or_path and not 'bnb' in args.model_name_or_path:
        if 'pangu' in args.model_name_or_path.lower():
            model_local_path = f"{PANGU_PATH}/{args.model_name_or_path}"


            # load the tokenizer and the model
            gen_tok = AutoTokenizer.from_pretrained(
                model_local_path, 
                use_fast=False, 
                trust_remote_code=True,
                local_files_only=True
            )

            gen_mdl = AutoModelForCausalLM.from_pretrained(
                model_local_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True
            )
            print(gen_mdl.device, gen_mdl)
        # if 'llama' in args.model_name_or_path.lower():
        #     gen_tok = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=True)
        else:
            gen_tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
            gen_mdl = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16,
            )
    else:
        gen_tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

        # 手动构造量化配置：4bit + groupsize=128 + act-order + true-sequential
        qcfg = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=True,          # 等价于 act-order
            true_sequential=True,   # 仓库卡片里写了 true-sequential
            sym=True                # 常用；不设置也行
        )

        gen_mdl = AutoGPTQForCausalLM.from_quantized(
            args.model_name_or_path,
            model_basename="llama-7b-4bit-128g",  # 必须与 .safetensors 前缀一致
            quantize_config=qcfg,                 # 关键：绕过缺失文件
            device_map="auto",
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_attention=False          # 某些环境关掉更稳
        )
    gen_mdl.eval()

    sim_mdl, sim_tok = load_simcse(args.sce_model_name, device=args.device)

    # ========== 5) 逐个目标攻击率：扰动 → 生成 → simcse → 去噪 → simcse ==========
    detail_records = []  # 若需保存每条样本的相似度（所有模型合并在一个文件里）
    for tar in sorted(search_results.keys()):
        eps = float(search_results[tar]["best_eps"])

        print(f"\n===== Eval for Target={tar:.3f} (ε={eps:.4f}) =====")
        perturbed_prompts, _, _, _, _ = perturb_sentences_batch_fast_with_counts(
            sentences=prompts_eval,
            epsilon=eps,
            resources=resources,
            origin_batch_size=args.origin_batch_size,
            numeric_strategy=args.numeric_strategy,
        )

        # before denoise: 只做一次（与去噪模型无关）
        gens_before = generate_with_causallm(gen_mdl, gen_tok, perturbed_prompts, max_new_tokens=args.max_new_tokens)
        sims_before = simcse_batch_cosine(gens_before, refs_eval, sim_mdl, sim_tok, batch_size=args.simcse_batch)
        sum_before = summarize_simcse("before-denoise", sims_before)
        search_results[tar]["simcse_before"] = float(sum_before["avg_similarity"])
        checkpoint_data = {
                'prompts_eval': prompts_eval,
                'perturbed_prompts': perturbed_prompts,
                'gens_before': gens_before,
                'target_rate': float(tar),
                'epsilon': float(eps),
                'metadata': {
                    'dataset': args.dataset,
                    'model': args.model_name_or_path,
                    'timestamp': str(datetime.now())
                }
            }

        checkpoint_dir = os.path.join('./tmp', "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_data_{args.dataset}_tar_{tar:.3f}_eps_{eps:.4f}.json"
        )

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        # after denoise: 对每个 GPTQ 模型分别进行
        for gm, gtag in zip(gptq_models, gptq_tags):
            print(f"\n--- Denoise using GPTQ model: {gm} ---")
            gptq_mdl, gptq_tok, gens_after = denoise_with_gptq(
                args.gptq,
                gm,
                list(zip(prompts_eval, gens_before)),
                max_new_tokens=args.max_new_tokens
            )
            sims_after = simcse_batch_cosine(gens_after, refs_eval, sim_mdl, sim_tok, batch_size=args.simcse_batch)
            sum_after = summarize_simcse(f"after-denoise ({gtag})", sims_after)

            # 记录该模型的平均相似度
            search_results[tar][f"simcse_after__{gtag}"] = float(sum_after["avg_similarity"])

            # 单独保存该模型对应的去噪文本（JSONL）
            denoise_dir = os.path.join(results_root, "denoise", gtag)

            auto_name = f"{safe_tag(args.dataset)}__{safe_tag(args.model_name_or_path)}__{gtag}__tar_{tar:.3f}__eps_{eps:.4f}.jsonl"
            save_denoised_jsonl(
                os.path.join(denoise_dir, auto_name),
                prompts=perturbed_prompts,
                gens_before=gens_before,
                gens_after=gens_after,
                refs=refs_eval,
                target_rate=tar,
                epsilon=eps,
                gptq_model=gm
            )

            # 细节合并写入（含 gptq_model 字段）
            if args.detail_json:
                for i in range(len(perturbed_prompts)):
                    detail_records.append({
                        "gptq_model": gm,
                        "gptq_tag": gtag,
                        "target_attack_rate": float(tar),
                        "epsilon": eps,
                        "prompt": perturbed_prompts[i],
                        "reference": refs_eval[i],
                        "gen_before": gens_before[i],
                        "gen_after": gens_after[i],
                        "sim_before": float(sims_before[i]),
                        "sim_after": float(sims_after[i]),
                    })

            # 释放该 GPTQ 模型显存
            del gptq_mdl
            del gptq_tok
            torch.cuda.empty_cache()

    print("\n===== Summary =====")
    for tar in sorted(search_results.keys()):
        beps = search_results[tar]["best_eps"]
        brate = search_results[tar]["attack_rate"]
        summary_line = f"Target={tar:.3f} => best ε={beps:.4f}, attack_rate={brate:.4f} | SimCSE before={search_results[tar]['simcse_before']:.4f}"
        for gtag in gptq_tags:
            v = search_results[tar].get(f"simcse_after__{gtag}", None)
            if v is not None:
                summary_line += f" | after({gtag})={v:.4f}"
        print(summary_line)

    # 自动构造 CSV 文件名：包含多个 GPTQ 模型 tag
    if args.csv is None:
        # 把 targets 转换成适合文件名的形式，比如 0.02 -> 0p02
        targets_str = "-".join([str(t).replace(".", "p") for t in args.targets])
        base = (
            f"{safe_tag(args.dataset)}__"
            f"{safe_tag(args.model_name_or_path)}__"
            f"{safe_tag('__'.join(args.gptq_model))}__"
            f"targets-{targets_str}.csv"
        )
        args.csv = os.path.join(results_root, base)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    export_results_to_csv(search_results, args.csv, gptq_tags=gptq_tags)

    # detail 合并保存（含 gptq_model 字段）
    if args.detail_json:
        os.makedirs(os.path.dirname(args.detail_json) or ".", exist_ok=True)
        with open(args.detail_json, "w", encoding="utf-8") as f:
            json.dump(detail_records, f, ensure_ascii=False)
        print(f"[export] detailed per-sample similarities saved to: {args.detail_json}")



if __name__ == "__main__":
    main()

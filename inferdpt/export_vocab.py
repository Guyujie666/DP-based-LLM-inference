import os, json, numpy as np, torch, argparse
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
PANGU_PATH = os.getenv("PANGU_PATH", "/default/pangu/path")
def safe_tag(model_name_or_path: str) -> str:
    # 仅包含字母数字和下划线，其它换成双下划线，便于跨平台文件名
    tag = model_name_or_path.strip().replace("\\", "/")
    for ch in ["/", ":", " ", "@", "#", "?", "&", "=", "+"]:
        tag = tag.replace(ch, "__")
    return tag

def find_embedding_module(
    model: torch.nn.Module,
    vocab_size: Optional[int] = None,
    debug: bool = False
) -> torch.nn.Embedding:
    """
    更鲁棒地定位输入嵌入层（nn.Embedding）：
    - 先用 HF 的 get_input_embeddings()
    - 再试常见路径（覆盖 Qwen2.5: transformer.embed_tokens）
    - 再兜底遍历，并做更宽松的匹配与打分
    """
    # 0) 取一些“参考尺寸”
    cfg_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if vocab_size is None:
        # 若外部没传 vocab_size，就尽量用 config.vocab_size
        vocab_size = cfg_vocab

    # 1) 直接用 HF 接口
    try:
        emb = getattr(model, "get_input_embeddings", None)
        if callable(emb):
            emb_layer = model.get_input_embeddings()
            if isinstance(emb_layer, torch.nn.Embedding):
                if debug:
                    print("[find_embedding_module] Found by get_input_embeddings()")
                return emb_layer
    except Exception as e:
        if debug:
            print(f"[find_embedding_module] get_input_embeddings() failed: {e}")

    # 2) 常见路径（覆盖多架构）
    common_paths = [
        "model.embed_tokens",             # LLaMA / 部分 Qwen
        "transformer.embed_tokens",       # ✅ Qwen2 / Qwen2.5
        "model.decoder.embed_tokens",     # OPT
        "transformer.wte",                # GPT-2
        "model.wte",
        "wte",
        "backbone.embed_tokens",          # 有些自定义权重会这么放
    ]
    for path in common_paths:
        cur = model
        ok = True
        for name in path.split("."):
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and isinstance(cur, torch.nn.Embedding):
            if debug:
                print(f"[find_embedding_module] Found by path: {path} "
                      f"(num_embeddings={cur.num_embeddings}, dim={cur.embedding_dim})")
            return cur

    # 3) 兜底遍历：收集所有候选，并按“更像输入嵌入”的程度打分
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Embedding):
            candidates.append((name, m))

    if debug:
        for name, m in candidates:
            print(f"[find_embedding_module] Candidate: {name}, "
                  f"num_embeddings={m.num_embeddings}, dim={m.embedding_dim}, dtype={m.weight.dtype}")

    if not candidates:
        raise RuntimeError("Cannot locate any nn.Embedding in model.")

    # 打分策略：
    #  - +3: m.num_embeddings == vocab_size
    #  - +3: m.num_embeddings == cfg_vocab
    #  - +2: m.embedding_dim == hidden_size
    #  - +1: 名称里包含 'embed'/'wte'
    def score(name_m):
        name, m = name_m
        s = 0
        if vocab_size is not None and m.num_embeddings == vocab_size:
            s += 3
        if cfg_vocab is not None and m.num_embeddings == cfg_vocab:
            s += 3
        if hidden_size is not None and m.embedding_dim == hidden_size:
            s += 2
        low = name.lower()
        if "embed" in low or "wte" in low:
            s += 1
        return s

    candidates.sort(key=score, reverse=True)
    best_name, best = candidates[0]

    if debug:
        print(f"[find_embedding_module] Picked: {best_name} "
              f"(num_embeddings={best.num_embeddings}, dim={best.embedding_dim}) "
              f"with score={score((best_name, best))}, "
              f"vocab_size={vocab_size}, cfg_vocab={cfg_vocab}, hidden_size={hidden_size}")

    return best
def export(model_name_or_path: str, out_dir: str, fp_dtype: str = "float32") -> Tuple[str,str,str]:
    os.makedirs(out_dir, exist_ok=True)
    tag = safe_tag(model_name_or_path)

    tok_path = os.path.join(out_dir, f"{tag}.tokens.json")
    emb_path = os.path.join(out_dir, f"{tag}.embeddings.npy")
    sen_path = os.path.join(out_dir, f"{tag}.sensitivity.npy")

    print(f"[load] tokenizer & model from: {model_name_or_path}")
    # ⚠️ 建议不要替换成固定的 LLaMA tokenizer，除非你确定必须这么做且缓存也用这个导出
    if 'pangu' in model_name_or_path.lower():
        model_local_path = f"{PANGU_PATH}/{model_name_or_path}"


        # load the tokenizer and the model
        tok = AutoTokenizer.from_pretrained(
            model_local_path, 
            use_fast=False, 
            trust_remote_code=True,
            local_files_only=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_local_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cuda",
            local_files_only=True
        )
    else:
        tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        # 尽量加载 CausalLM，不行再退 AutoModel
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float32, device_map=None
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_name_or_path, torch_dtype=torch.float32, device_map=None
            )
    model.eval()

    # === 关键：以模型嵌入的行数为准 ===
    emb_layer = find_embedding_module(model, vocab_size=None, debug=False)
    with torch.no_grad():
        W = emb_layer.weight.detach().cpu()
        if W.dtype == torch.bfloat16:
            W = W.to(torch.float32)
        W = W.numpy()
    V, d = W.shape
    print(f"[info] embed rows (V) = {V}, dim = {d}, len(tokenizer) = {len(tok)}, tokenizer.vocab_size = {getattr(tok, 'vocab_size', None)}")

    # 1) tokens.json：严格导出 0..V-1 的 id → token
    tokens = []
    for i in range(V):
        tok_i = tok.convert_ids_to_tokens(i)
        if tok_i is None:
            tok_i = f"<ID_{i}>"
        tokens.append(tok_i)
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False)
    print(f"[ok] wrote tokens -> {tok_path} (len={len(tokens)})")

    # 2) embeddings.npy：与 tokens 行数一致
    if fp_dtype == "float32":
        W = W.astype(np.float32, copy=False)
    elif fp_dtype == "float64":
        W = W.astype(np.float64, copy=False)
    else:
        raise ValueError("fp_dtype must be float32 or float64")
    np.save(emb_path, W)
    print(f"[ok] wrote embeddings -> {emb_path}  shape={W.shape}")

    # 3) sensitivity.npy：逐维 (max-min)
    delta = (W.max(axis=0).astype(np.float64) - W.min(axis=0).astype(np.float64))
    np.save(sen_path, delta)
    print(f"[ok] wrote sensitivity -> {sen_path}  shape={delta.shape}")


    assert len(tokens) == W.shape[0], "tokens length must equal embeddings rows"
    return tok_path, emb_path, sen_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g., meta-llama/Llama-3-8B, facebook/opt-6.7b, gpt2")
    ap.add_argument("--out_dir", default="./")
    ap.add_argument("--dtype", default="float32", choices=["float32","float64"])
    args = ap.parse_args()
    export(args.model, args.out_dir, args.dtype)

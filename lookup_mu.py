#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, sys
from pathlib import Path

def safe_tag(s: str) -> str:
    s = s.strip().replace("\\", "/")
    for ch in ["/", ":", " ", "@", "#", "?", "&", "=", "+"]:
        s = s.replace(ch, "__")
    return s

def _str_key(x):
    if isinstance(x, float):
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return str(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--mechanism", required=True, help="如 ternary / gaussian / chidp")
    p.add_argument("--quant_level", type=int, required=True)
    p.add_argument("--targets", type=float, nargs="+", required=True, help="目标攻击成功率列表，顺序很重要")
    args = p.parse_args()

    safe_model = safe_tag(args.model)
    fname = f"{args.dataset}__{safe_model}__proj{args.proj_dim}__{args.mechanism.lower()}.json"
    fpath = Path(args.output_dir) / fname
    if not fpath.exists():
        print(f"[lookup_mu] file not found: {fpath}", file=sys.stderr)
        sys.exit(2)

    with open(fpath, "r") as f:
        payload = json.load(f)

    table = payload.get("table", {})
    qkey = _str_key(args.quant_level)
    qtbl = table.get(qkey, {})
    if not qtbl:
        print(f"[lookup_mu] quant_level={args.quant_level} not found in {fpath}", file=sys.stderr)
        sys.exit(3)

    mus = []
    for ta in args.targets:
        tkey = _str_key(ta)
        row = qtbl.get(tkey)
        if not row or "error" in row:
            print(f"[lookup_mu] missing param for target_attack_rate={ta} (quant={args.quant_level})", file=sys.stderr)
            sys.exit(4)
        if str(row.get("param_name", "")).lower() != "mu":
            print(f"[lookup_mu] param_name is {row.get('param_name')} not mu", file=sys.stderr)
            sys.exit(5)
        mus.append(str(row["param_value"]))

    # 空格分隔打印，便于 bash 捕获为数组
    print(" ".join(mus))

if __name__ == "__main__":
    main()

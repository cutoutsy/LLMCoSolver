#!/usr/bin/env python
"""把 LoRA 适配器合并进基座模型，产出可直接评测的完整模型目录。

用法：
    python merge_lora.py output_hf_tsp/checkpoint-500 --out saved_models
    python merge_lora.py <ckpt_dir> --out saved_models --dtype bfloat16
"""
import argparse
import os
import shutil

import torch
from peft import AutoPeftModelForCausalLM

# 训练产物，不复制到合并结果里
SKIP_PREFIX = (
    "adapter_model", "adapter_config", "optimizer.pt", "scheduler.pt",
    "rng_state", "trainer_state", "training_args", "README.md", "logs",
)


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    ap.add_argument("ckpt", help="LoRA checkpoint 目录（含 adapter_config.json）")
    ap.add_argument("--out", default="saved_models", help="输出目录")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--keep_existing", action="store_true", help="不清空已有输出目录")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.ckpt, "adapter_config.json")):
        raise SystemExit(f"{args.ckpt} 里没有 adapter_config.json，不是 LoRA checkpoint")

    if os.path.isdir(args.out) and not args.keep_existing:
        print(f"清空已有目录 {args.out}/")
        shutil.rmtree(args.out)

    dtype = getattr(torch, args.dtype)
    print(f"加载 LoRA: {args.ckpt}  (dtype={args.dtype})")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    print("合并 LoRA 到基座 ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="2GB")
    print(f"合并后模型已保存到 {args.out}/")

    # eval.py 会用 AutoTokenizer.from_pretrained(model_id)，所以 tokenizer 文件必须一起带过去
    copied = []
    for f in sorted(os.listdir(args.ckpt)):
        if f.startswith(SKIP_PREFIX):
            continue
        src = os.path.join(args.ckpt, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, f))
            copied.append(f)
    print(f"已复制 {len(copied)} 个附属文件: {', '.join(copied)}")
    print("\n下一步评测：")
    print(f"  python eval.py --model_id {args.out} --problem tsp --eval_method vanilla --num_samples 20")


if __name__ == "__main__":
    main()

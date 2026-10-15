#!/usr/bin/env python
"""把「漂亮打印的 JSON 数组」转成 JSONL（每行一个对象），供 datasets 流式加载。

背景：
  官方 data/tsp/train/train_tsp.json 是 json.dump(..., indent=4) 产出的 3.45GB 数组。
  datasets 3.6.0 的 JSON 加载器对它两条路都失败：
    ① pyarrow 分块读只认 JSONL → "Column() changed from object to array in row 0"
    ② pandas 回退把整文件读成 pyarrow-backed DataFrame，字符串列 > 2GB
       → "Failed casting from large_string to string: input array too large"
  转成 JSONL 后，① 这条流式路径即可正常工作，且不需要改任何项目代码
  （目录名 train/ + 文件名 train.jsonl 依然能被 split="train" 命中）。

用法：
    python json_to_jsonl.py data/tsp/train/train_tsp.json          # 生成同名 .jsonl
    python json_to_jsonl.py <src> <dst.jsonl> [--expect N]
"""
import argparse
import json
import os
import sys
import time


def convert(src: str, dst: str, expect: int | None = None) -> int:
    n = 0
    buf: list[str] = []
    t0 = time.time()
    size = os.path.getsize(src)
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            s = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            if not buf:                            # 记录之外：只可能是外层数组的括号
                if not s or s == "[" or s == "]":
                    continue
                buf.append(line)
                continue
            buf.append(line)
            # 记录结束的判据 = 「缩进恰好 4 空格」+ 内容为 } 或 },
            # （嵌套结构的闭合符缩进 ≥8，例如 "instance" 里的 [ ] 与内层对象）
            if indent == 4 and s in ("}", "},"):
                chunk = "".join(buf).rstrip()
                if chunk.endswith(","):
                    chunk = chunk[:-1]
                rec = json.loads(chunk)            # 逐条校验，坏数据立刻暴露
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                buf = []
                if n % 50000 == 0:
                    el = time.time() - t0
                    rate = n / el
                    eta = f"，预计剩余 {((expect - n) / rate) / 60:.1f} 分钟" if expect else ""
                    print(f"  {n:>7,} 条  ({el:5.1f}s, {rate:,.0f} 条/秒{eta})", flush=True)
    if buf:
        raise RuntimeError(f"文件似乎被截断：最后还有 {len(buf)} 行未闭合")
    print(f">> {dst}: {n:,} 条, {os.path.getsize(dst)/2**30:.2f} GiB, "
          f"耗时 {time.time()-t0:.1f}s")
    if expect is not None and n != expect:
        raise RuntimeError(f"条数不符：得到 {n}，期望 {expect}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--expect", type=int, default=None, help="期望条数（用于校验）")
    a = ap.parse_args()
    dst = a.dst or os.path.splitext(a.src)[0] + ".jsonl"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        sys.exit(f"目标已存在，先删除：{dst}")
    print(f"转换 {a.src} -> {dst}")
    convert(a.src, dst, a.expect)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""量化误差深挖：逐层反量化 vs fp32，定位精度损失来源与 embedding 异常行。"""
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = r"D:\DEVELOP\Project\ai8_-project1\models\bert_distillation_quantization\model"
fp32 = torch.load(f"{ROOT}/student_bert_4l384.pt", map_location="cpu", weights_only=True)
qs = torch.load(f"{ROOT}/student_bert_4l384_int8.pt", map_location="cpu", weights_only=True)

print("=" * 92)
print("【1】把 int8 里被 tuple 包住的 Linear 权重全部解出来")
print("=" * 92)
pairs = {}          # 模块前缀 -> (deq_weight, fp32_weight, bias_fp32)
elem_in_tuples = 0
for k, v in qs.items():
    if not isinstance(v, tuple):
        continue
    prefix = k.replace("._packed_params._packed_params", "")
    w = v[0]
    b = v[1] if len(v) > 1 else None
    if not torch.is_tensor(w):
        print(f"  !! {prefix} 的权不是张量: {type(w)}")
        continue
    elem_in_tuples += w.numel() + (b.numel() if torch.is_tensor(b) else 0)
    pairs[prefix] = (w, b)
print(f"  解出 Linear 模块数: {len(pairs)}")
print(f"  tuple 内元素总数  : {elem_in_tuples:,}")

total_int8 = sum(v.numel() for v in qs.values() if torch.is_tensor(v)) + elem_in_tuples
total_fp32 = sum(v.numel() for v in fp32.values() if torch.is_tensor(v))
print(f"  int8 总元素(修正后): {total_int8:,}")
print(f"  fp32 总元素        : {total_fp32:,}")
print(f"  完全一致           : {total_int8 == total_fp32}")
missing = set(fp32) - set(qs)
print(f"  fp32 有而 int8 没有的独立键: {len(missing)} 个（应全部落在 tuple 内）")
print(f"    其中仍是张量形式残留的: "
      f"{[m for m in missing if any(m.endswith(s) for s in ('LayerNorm.weight','LayerNorm.bias'))]}")

print("\n" + "=" * 92)
print("【2】逐模块量化误差（反量化 vs fp32）")
print("=" * 92)
print(f"  {'模块':<52}{'相对误差':>10}{'余弦相似':>11}{'最大绝对误差':>12}")
rows_err = []
for k in sorted(pairs):
    wq, bq = pairs[k]
    w32 = fp32.get(k + ".weight")
    if w32 is None:
        continue
    deq = wq.dequantize().to(torch.float32) if wq.is_quantized else wq.to(torch.float32)
    if deq.shape != w32.shape:
        print(f"  !! {k}: shape 不一致 {tuple(deq.shape)} vs {tuple(w32.shape)}")
        continue
    diff = (deq - w32).flatten()
    rel = diff.norm() / w32.flatten().norm()
    cos = torch.nn.functional.cosine_similarity(
        deq.flatten().unsqueeze(0), w32.flatten().unsqueeze(0)).item()
    maxabs = diff.abs().max().item()
    rows_err.append((k, float(rel), cos, maxabs))
for k, rel, cos, mx in sorted(rows_err, key=lambda x: -x[1])[:30]:
    print(f"  {k:<52}{rel:>10.3%}{cos:>11.5f}{mx:>12.2e}")

rel_all = [r[1] for r in rows_err]
print(f"\n  平均相对误差 {np.mean(rel_all):.3%}   最差 {max(rel_all):.3%}   最好 {min(rel_all):.3%}")

print("\n" + "=" * 92)
print("【3】词嵌入（占 52% 参数）逐行检查")
print("=" * 92)
emb32 = fp32["bert.embeddings.word_embeddings.weight"]
emb_q = qs["bert.embeddings.word_embeddings._packed_params._packed_weight"]
scales = emb_q.q_per_channel_scales().to(torch.float32)
zps = emb_q.q_per_channel_zero_points().to(torch.float32)
deq_emb = emb_q.dequantize().to(torch.float32)
row_range32 = emb32.abs().max(dim=1).values

print(f"  fp32 行最大绝对值: 中位 {row_range32.median():.4f}  最大 {row_range32.max():.4f}")
print(f"  量化 scale        : 中位 {scales.median():.3e}  最小 {scales.min():.3e}  最大 {scales.max():.3e}")

degen = (scales == 1.0)
print(f"\n  scale == 1.0 的行数: {int(degen.sum())} / {scales.numel()}")
if degen.any():
    idx = degen.nonzero().flatten()
    print(f"    这些行在 fp32 中的最大绝对值: 全部=0 ? {bool((row_range32[idx] == 0).all())}")
    print(f"    样例 5 行 fp32 最大值: {row_range32[idx][:5].tolist()}")

zeroed = (deq_emb.abs().max(dim=1).values == 0) & (row_range32 > 0)
print(f"  ★ 行在 fp32 非零、但反量化后变成全 0 的: {int(zeroed.sum())} 行")
if zeroed.any():
    zidx = zeroed.nonzero().flatten()
    print(f"    这些行 fp32 最大绝对值 中位={row_range32[zidx].median():.3e} 最大={row_range32[zidx].max():.3e}")

# 用词表看这些行是什么 token
vocab = open(r"D:\DEVELOP\Project\ai8_-project1\models\bert_distillation_quantization\bert-base-chinese\vocab.txt",
             encoding="utf-8").read().splitlines()
print(f"\n  词表大小: {len(vocab)}  (嵌入行数 {emb32.shape[0]})")
zs = zeroed.nonzero().flatten().tolist()
if zs:
    print(f"  被清零的行前 20 个 token: {[repr(vocab[i]) if i < len(vocab) else i for i in zs[:20]]}")

# 词向量的整体误差
d = (deq_emb - emb32)
print(f"\n  词嵌入整体相对误差: {d.norm() / emb32.norm():.3%}")
print(f"  词嵌入逐行误差 中位: {(d.norm(dim=1) / emb32.norm(dim=1)).median():.3%}")

print("\n" + "=" * 92)
print("【4】int8 文件里各部分的实际体积占比")
print("=" * 92)
def nbytes(t):
    if t.is_quantized:
        return t.int_repr().numel() + 4 * (t.q_per_channel_scales().numel() if t.qscheme() in
                                           (torch.per_channel_affine, torch.per_channel_affine_float_qparams) else 1) \
               + 8 * (t.q_per_channel_zero_points().numel() if t.qscheme() in
                      (torch.per_channel_affine_float_qparams,) else 1)
    return t.numel() * t.element_size()

bucket = {}
for k, v in qs.items():
    if isinstance(v, tuple):
        for j, t in enumerate(v):
            if torch.is_tensor(t):
                bucket.setdefault("Linear(打包)", 0)
                bucket["Linear(打包)"] += nbytes(t)
        continue
    if not torch.is_tensor(v):
        continue
    if "word_embeddings" in k:
        bucket["Embedding", 0]
        bucket["Embedding"] = bucket.get("Embedding", 0) + nbytes(v)
    elif "position_embeddings" in k or "token_type_embeddings" in k:
        bucket["Embedding(位置/类型)", 0]
        bucket["Embedding(位置/类型)"] = bucket.get("Embedding(位置/类型)", 0) + nbytes(v)
    elif "LayerNorm" in k:
        bucket["LayerNorm(未量化)", 0]
        bucket["LayerNorm(未量化)"] = bucket.get("LayerNorm(未量化)", 0) + nbytes(v)
    else:
        bucket["量化标量/其他", 0]
        bucket["量化标量/其他"] = bucket.get("量化标量/其他", 0) + nbytes(v)
tot = sum(bucket.values())
for k, b in sorted(bucket.items(), key=lambda x: -x[1]):
    print(f"  {k:<24}{b / 1024**2:>9.3f} MiB  {b / tot:>7.2%}")
print(f"  {'合计(按张量)':<24}{tot / 1024**2:>9.3f} MiB")
print(f"  文件实际大小                             15.12 MiB")

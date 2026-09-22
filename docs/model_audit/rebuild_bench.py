# -*- coding: utf-8 -*-
"""端到端实测：
  1) 能否从两个 .pt 干净重建模型（结构化完整性）
  2) 在真实 val.csv 上复现 eval_result.txt 的 macro-F1
  3) 不同 padding / 批大小策略下的 CPU 推理性能
不 import 项目里的 bert_config（它会在缺基座权重时崩溃），自己按 config.json 重建同构学生。
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT = r"D:\DEVELOP\Project\ai8_-project1"
MOD = ROOT + r"\models\bert_distillation_quantization"
DATA = ROOT + r"\data\processed\final_data"

import transformers

torch.backends.quantized.engine = "onednn"
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("量化引擎", torch.backends.quantized.engine,
      "| 可用", torch.backends.quantized.supported_engines)
print("CPU 线程数 torch.get_num_threads() =", torch.get_num_threads())
print("机器逻辑核心 os.cpu_count() =", os.cpu_count())


# ----------------------------------------------------------------- 1. 重建架构
class StudentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = transformers.BertConfig.from_pretrained(MOD + r"\bert-base-chinese")
        cfg = transformers.BertConfig(
            vocab_size=base.vocab_size, hidden_size=384, num_hidden_layers=4,
            num_attention_heads=6, intermediate_size=1536, hidden_act=base.hidden_act,
            hidden_dropout_prob=base.hidden_dropout_prob,
            attention_probs_dropout_prob=base.attention_probs_dropout_prob,
            max_position_embeddings=base.max_position_embeddings,
            type_vocab_size=base.type_vocab_size, initializer_range=base.initializer_range,
            layer_norm_eps=base.layer_norm_eps, pad_token_id=base.pad_token_id)
        self.bert = transformers.BertModel(cfg)
        self.cat_linear = torch.nn.Linear(384, 7)
        self.sent_linear = torch.nn.Linear(384, 2)

    def forward(self, x):
        r = self.bert(**x)
        p = r["pooler_output"]
        return self.cat_linear(p), self.sent_linear(p)


print("\n" + "=" * 90)
print("【1】从 .pt 重建模型（结构化完整性检查）")
print("=" * 90)

m32 = StudentModel()
sd32 = torch.load(MOD + r"\model\student_bert_4l384.pt", map_location="cpu", weights_only=True)
miss32, unexp32 = m32.load_state_dict(sd32, strict=True), None
m32.eval()
n32 = sum(p.numel() for p in m32.parameters())
print(f"  fp32  strict=True 加载通过 | 参数量 {n32:,}")

from torch.ao.quantization import (default_dynamic_qconfig, float_qparams_weight_only_qconfig,
                                   quantize_dynamic)

qcfg = {torch.nn.Linear: default_dynamic_qconfig,
        torch.nn.Embedding: float_qparams_weight_only_qconfig}
m8 = quantize_dynamic(StudentModel(), qconfig_spec=qcfg, dtype=torch.qint8)
sd8 = torch.load(MOD + r"\model\student_bert_4l384_int8.pt", map_location="cpu", weights_only=True)
r8 = m8.load_state_dict(sd8, strict=True)
m8.eval()
print(f"  int8  strict=True 加载通过")
print(f"  -> 两个权重文件都能与代码里的结构**严格**对齐，无缺失/多余键")

# 反量化对照：把量化权重灌回去是否等价
print("\n  int8 反量化模型（把 int8 权重解回 fp32）构造中 ...")
sd8_deq = {}
for k, v in sd8.items():
    if isinstance(v, tuple):
        sd8_deq[k] = tuple(t.dequantize() if torch.is_tensor(t) and t.is_quantized else t for t in v)
    elif torch.is_tensor(v) and v.is_quantized:
        sd8_deq[k] = v.dequantize()
    else:
        sd8_deq[k] = v

# ----------------------------------------------------------------- 2. 真实数据评估
print("\n" + "=" * 90)
print("【2】真实 val.csv 上复现指标")
print("=" * 90)
val = pd.read_csv(DATA + r"\val.csv", encoding="utf-8-sig")
print(f"  val 条数: {len(val)}   列: {list(val.columns)}")
class_txt = open(DATA + r"\class.txt", encoding="utf-8").read().splitlines()
cat2id = {c: i for i, c in enumerate(class_txt)}
y_cat = np.array([cat2id[str(x)] for x in val["cat_l1"]])
y_sent = val["label"].to_numpy()
texts = [str(t) for t in val["review"]]

tok = transformers.BertTokenizer.from_pretrained(MOD + r"\bert-base-chinese")


def run_eval(model, batch_size=64, max_len=256, dynamic=False, sort_len=False, limit=None):
    model.eval()
    idx = np.arange(len(texts))
    if limit:
        idx = idx[:limit]
    if sort_len:
        # 先按 token 长度排序分桶（与评估指标无关，只影响速度）
        lens = np.array([len(tok.encode(texts[i], truncation=False)) for i in idx])
        idx = idx[np.argsort(lens)]
    pc, ps = [], []
    t0 = time.time()
    total_tokens = 0
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            chunk = [texts[i] for i in idx[s:s + batch_size]]
            enc = tok(chunk, max_length=max_len, padding="max_length" if not dynamic else "longest",
                      truncation=True, return_tensors="pt")
            total_tokens += int(enc["attention_mask"].sum())
            c, se = model(enc)
            pc.extend(torch.argmax(c, -1).tolist())
            ps.extend(torch.argmax(se, -1).tolist())
    el = time.time() - t0
    cf = f1_score(y_cat[idx], pc, average="macro", zero_division=0)
    sf = f1_score(y_sent[idx], ps, average="macro", zero_division=0)
    return dict(sec=el, n=len(idx), ms_per=el / len(idx) * 1000,
                cat_f1=cf, sent_f1=sf, avg=(cf + sf) / 2,
                tokens_per_s=total_tokens / el)


print("\n  --- 全量 val（9390 条），与 eval_result.txt 对齐 ---")
print(f"  {'配置':<34}{'总耗时s':>10}{'ms/条':>10}{'大类F1':>10}{'情感F1':>10}{'综合F1':>10}")
ref = {"fp32": (0.8943, 0.9202, 0.9072), "int8": (0.8943, 0.9204, 0.9074)}
store = {}
for tag, model in (("fp32", m32), ("int8", m8)):
    r = run_eval(model, 64, 256, dynamic=False)
    store[tag] = r
    e = ref[tag]
    print(f"  {tag+' 固定padding(现状, batch64)':<34}{r['sec']:>10.1f}{r['ms_per']:>10.2f}"
          f"{r['cat_f1']:>10.4f}{r['sent_f1']:>10.4f}{r['avg']:>10.4f}")
    print(f"  {'  eval_result.txt 记录值':<34}{'':>10}{'':>10}"
          f"{e[0]:>10.4f}{e[1]:>10.4f}{e[2]:>10.4f}")
    print(f"  {'  是否复现(容差5e-4)':<34}"
          f"{'':>10}{'':>10}{'是' if abs(r['cat_f1']-e[0])<5e-4 else '否':>10}"
          f"{'是' if abs(r['sent_f1']-e[1])<5e-4 else '否':>10}"
          f"{'是' if abs(r['avg']-e[2])<5e-4 else '否':>10}")

# ----------------------------------------------------------------- 3. 性能优化对比
print("\n" + "=" * 90)
print("【3】输入长度策略对 CPU 推理性能的影响（val 前 1024 条，batch=64）")
print("=" * 90)
LIM = 1024
sub = np.array([len(tok.encode(texts[i], truncation=False)) for i in range(LIM)])
print(f"  子集 token 长度: 中位={int(np.median(sub))} 均值={sub.mean():.1f} "
      f"P90={int(np.percentile(sub,90))} 最大={int(sub.max())}")
print(f"  现状做法 padding 到 256 -> 平均约 {256} 个位置参与计算，实际有效仅 {sub.mean():.1f} 个 "
      f"(浪费 {1 - sub.mean()/256:.1%})\n")
print(f"  {'方案':<40}{'ms/条':>9}{'加速':>8}{'有效token/s':>14}")
base_ms = None
for tag, model in (("fp32", m32), ("int8", m8)):
    print(f"  -- {tag} --")
    combos = [
        ("A 固定256 + batch64 (现状)", dict(batch_size=64, max_len=256, dynamic=False)),
        ("B 动态padding + batch64", dict(batch_size=64, max_len=256, dynamic=True)),
        ("C 动态padding+按长度分桶 + batch64", dict(batch_size=64, max_len=256, dynamic=True, sort_len=True)),
        ("D 动态padding+分桶 + batch32", dict(batch_size=32, max_len=256, dynamic=True, sort_len=True)),
        ("E 动态padding+分桶 + max_len=128", dict(batch_size=64, max_len=128, dynamic=True, sort_len=True)),
    ]
    b = None
    for name, kw in combos:
        r = run_eval(model, limit=LIM, **kw)
        if b is None:
            b = r["ms_per"]
        print(f"  {name:<40}{r['ms_per']:>9.2f}{b / r['ms_per']:>7.2f}x{r['tokens_per_s']:>14.0f}")

print("\n" + "=" * 90)
print("【4】fp32 vs int8 一致率 + 反量化模型交叉验证")
print("=" * 90)
rm32 = run_eval(m32, limit=LIM)
rm8 = run_eval(m8, limit=LIM)
print(f"  fp32 综合F1(子集)={rm32['avg']:.4f}   int8 综合F1(子集)={rm8['avg']:.4f}  "
      f"差={rm8['avg']-rm32['avg']:+.4f}")
print(f"  fp32 {rm32['ms_per']:.2f} ms/条 vs int8 {rm8['ms_per']:.2f} ms/条 "
      f"-> int8 加速 {rm32['ms_per']/rm8['ms_per']:.2f}x")
print("\n  完成")

# -*- coding: utf-8 -*-
"""性能细拆：把「分词」「前向」分开计时，并做充分 warmup + 多次取最小值，避免噪声。
同时测进程内存占用（权重 + 推理峰值）。"""
import ctypes
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import transformers
from torch.ao.quantization import (default_dynamic_qconfig, float_qparams_weight_only_qconfig,
                                   quantize_dynamic)

warnings.filterwarnings("ignore")
torch.backends.quantized.engine = "onednn"

ROOT = r"D:\DEVELOP\Project\ai8_-project1"
MOD = ROOT + r"\models\bert_distillation_quantization"
DATA = ROOT + r"\data\processed\final_data"


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
        p = self.bert(**x)["pooler_output"]
        return self.cat_linear(p), self.sent_linear(p)


def rss_mb():
    """当前进程 RSS（Windows）。"""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    p = PMC(); p.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                             ctypes.byref(p), p.cb)
    return p.WorkingSetSize / 1024**2, p.PeakWorkingSetSize / 1024**2


m32 = StudentModel()
m32.load_state_dict(torch.load(MOD + r"\model\student_bert_4l384.pt", map_location="cpu",
                               weights_only=True), strict=True)
m32.eval()
qcfg = {torch.nn.Linear: default_dynamic_qconfig,
        torch.nn.Embedding: float_qparams_weight_only_qconfig}
m8 = quantize_dynamic(StudentModel(), qconfig_spec=qcfg, dtype=torch.qint8)
m8.load_state_dict(torch.load(MOD + r"\model\student_bert_4l384_int8.pt", map_location="cpu",
                              weights_only=True), strict=True)
m8.eval()

tok = transformers.BertTokenizer.from_pretrained(MOD + r"\bert-base-chinese")
val = pd.read_csv(DATA + r"\val.csv", encoding="utf-8-sig")
texts = [str(t) for t in val["review"][:1024]]
print(f"线程数 {torch.get_num_threads()} / 逻辑核 {os.cpu_count()}")
cur, peak = rss_mb()
print(f"两个模型都加载后 RSS = {cur:.0f} MiB (峰值 {peak:.0f} MiB)\n")

N, BS = 1024, 64


def bench(pad_mode, model=None, model_tag="", repeats=3, warmup=2):
    """pad_mode: 'fixed256' | 'bucket'
    返回 (分词 ms/条, 前向 ms/条)"""
    order = np.arange(N)
    if pad_mode == "bucket":
        lens = np.array([len(tok.encode(texts[i], truncation=False)) for i in order])
        order = order[np.argsort(lens)]
    batches = []
    for s in range(0, N, BS):
        chunk = [texts[i] for i in order[s:s + BS]]
        enc = tok(chunk, max_length=256,
                  padding="max_length" if pad_mode == "fixed256" else "longest",
                  truncation=True, return_tensors="pt")
        batches.append(enc)

    tok_ms = None
    if model is not None:
        tok_t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for s in range(0, N, BS):
                tok([texts[i] for i in order[s:s + BS]], max_length=256,
                    padding="max_length" if pad_mode == "fixed256" else "longest",
                    truncation=True, return_tensors="pt")
            tok_t.append(time.perf_counter() - t0)
        tok_ms = min(tok_t) / N * 1000

    with torch.no_grad():
        for _ in range(warmup):
            for enc in batches:
                model(enc)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for enc in batches:
                model(enc)
            ts.append(time.perf_counter() - t0)
    return tok_ms, min(ts) / N * 1000


print("=" * 96)
print("【A】分词成本 vs 模型前向成本（val 前 1024 条，batch=64，取 3 次最小值）")
print("=" * 96)
print(f"{'方案':<30}{'分词 ms/条':>12}{'前向 ms/条':>12}{'合计 ms/条':>12}{'前向占比':>10}")
res = {}
for pad, name in (("fixed256", "现状：固定 padding 到 256"), ("bucket", "优化：动态padding+按长度分桶")):
    for tag, mdl in (("fp32", m32), ("int8", m8)):
        t, f = bench(pad, mdl)
        res[(pad, tag)] = (t, f)
        print(f"  {name+'  '+tag:<28}{t:>12.2f}{f:>12.2f}{t+f:>12.2f}{f/(t+f):>9.1%}")
    print()

print("=" * 96)
print("【B】结论性对比")
print("=" * 96)
tA_f, fA_f = res[("fixed256", "fp32")]
tA_8, fA_8 = res[("fixed256", "int8")]
tB_f, fB_f = res[("bucket", "fp32")]
tB_8, fB_8 = res[("bucket", "int8")]
print(f"  1) 单看模型前向，int8 相对 fp32 加速:")
print(f"      固定padding: {fA_f/fA_8:.2f}x     优化padding: {fB_f/fB_8:.2f}x")
print(f"  2) 端到端（含分词）int8 相对 fp32 加速:")
print(f"      固定padding: {(tA_f+fA_f)/(tA_8+fA_8):.2f}x     优化padding: {(tB_f+fB_f)/(tB_8+fB_8):.2f}x")
print(f"  3) 仅改输入长度策略（不动模型）带来的端到端加速:")
print(f"      fp32: {(tA_f+fA_f)/(tB_f+fB_f):.2f}x        int8: {(tA_8+fA_8)/(tB_8+fB_8):.2f}x")
print(f"  4) 分词成本占总耗时: fp32-现状 {tA_f/(tA_f+fA_f):.1%} / int8-现状 {tA_8/(tA_8+fA_8):.1%}")
print(f"      -> 分词是固定开销，量化不会让它变快，因此它稀释了 int8 的加速比")
print(f"  5) 两项一起用（int8 + 分桶）相对现状 fp32:")
print(f"      {(tA_f+fA_f)/(tB_8+fB_8):.2f}x")
print()
cur, peak = rss_mb()
print(f"  进程 RSS 现在 {cur:.0f} MiB，峰值 {peak:.0f} MiB")

# ============================================
# max_len 决策分析：用真实 BERT tokenizer 计算 token 长度分布
# 目的：确定 config.py 里 max_len 该设多少，覆盖多少比例的评论
# ============================================
import pandas as pd
import numpy as np
import transformers
import json
import os

# ---------- 配置 ----------
# 项目根从本文件位置推算（本文件在 models/bert/ 下，向上三级即项目根）。
# 既避免依赖运行时的 cwd，也不再写死某台机器的绝对路径。
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE, "data", "processed", "final_data", "train.csv")
OUT_JSON = os.path.join(BASE, "data", "processed", "final_data", "maxlen_stats.json")
# 复用已下载好的中文 BERT tokenizer（最准）。按优先级找：
#   ① 本模块目录 models/bert/bert-base-chinese
#   ② 蒸馏模块目录（它的 config/tokenizer 已入库，任何机器 clone 后都有）
#   ③ 都没有才回退到 HuggingFace 仓库 id
BERT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert-base-chinese")
if not os.path.exists(os.path.join(BERT_PATH, "config.json")):
    BERT_PATH = os.path.join(BASE, "models", "bert_distillation_quantization", "bert-base-chinese")
if not os.path.exists(os.path.join(BERT_PATH, "config.json")):
    BERT_PATH = "bert-base-chinese"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")   # 数据 CSV 带 BOM，必须显式指定
print(f"读入数据：{len(df)} 条")

# ---------- 1. 字符长度分布（作为参考）----------
df['char_len'] = df['review'].astype(str).str.len()

# ---------- 2. 真实 token 长度分布 ----------
print("加载 BERT tokenizer:", BERT_PATH)
tok = transformers.BertTokenizer.from_pretrained(BERT_PATH)

def tok_len(text):
    # 不加特殊符，只看正文 token 数；BERT 实际会 +2（[CLS]/[SEP]）
    return len(tok.tokenize(str(text)))

df['token_len'] = df['review'].apply(tok_len)
# BERT 实际输入长度 = token_len + 2（[CLS] 和 [SEP]）
df['bert_input_len'] = df['token_len'] + 2

# ---------- 3. 分位数 ----------
print("\n=== 字符长度（字）分位数 ===")
for q in [0.5, 0.9, 0.95, 0.99, 1.0]:
    print(f"  P{int(q*100):>2}: {df['char_len'].quantile(q):.0f}")

print("\n=== BERT 真实输入长度（token+2）分位数 ===")
for q in [0.5, 0.9, 0.95, 0.99, 1.0]:
    print(f"  P{int(q*100):>2}: {df['bert_input_len'].quantile(q):.0f}")

# ---------- 4. 不同 max_len 的截断比例 ----------
print("\n=== 各候选 max_len 的截断情况（超过该长度会被截断）===")
MAX_BERT = 512
candidates = [32, 64, 128, 256, 512]
overflow = {}
for m in candidates:
    n_over = int((df['bert_input_len'] > m).sum())
    pct = n_over / len(df) * 100
    overflow[str(m)] = {"truncated": n_over, "pct": round(pct, 3)}
    print(f"  max_len={m:>3}: 被截 {n_over:>5} 条 ({pct:5.2f}%)  保留 {(len(df)-n_over):>5} 条")

# ---------- 5. 直方图数据（供绘图/决策）----------
bins = list(range(0, 270, 10))
hist, edges = np.histogram(df['bert_input_len'].clip(upper=260), bins=bins)
hist_data = {
    "bin_start": [int(edges[i]) for i in range(len(edges)-1)],
    "bin_end": [int(edges[i+1]) for i in range(len(edges)-1)],
    "count": [int(h) for h in hist],
}
# 把关键统计也汇总成一段给 jev 当 state 用的文字
stats_text = (
    f"训练集共 {len(df)} 条中文商品评论。\n"
    f"BERT 真实输入长度（正文 token 数 + [CLS]/[SEP] 共 2 个特殊符）统计：\n"
    f"中位数={df['bert_input_len'].quantile(0.5):.0f}，"
    f"P90={df['bert_input_len'].quantile(0.9):.0f}，"
    f"P95={df['bert_input_len'].quantile(0.95):.0f}，"
    f"P99={df['bert_input_len'].quantile(0.99):.0f}，"
    f"最大值={df['bert_input_len'].max():.0f}。\n"
    f"候选 max_len 的截断比例："
    f"32→{overflow['32']['pct']}%，64→{overflow['64']['pct']}%，"
    f"128→{overflow['128']['pct']}%，256→{overflow['256']['pct']}%，"
    f"512→{overflow['512']['pct']}%。\n"
    f"BERT 绝对上限为 512。max_len 越大，batch 内 padding 越多、显存与计算开销越大；"
    f"过小则会截断长评论丢失信息。"
)

summary = {
    "stats_text": stats_text,
    "percentiles": {
        "P50": float(df['bert_input_len'].quantile(0.5)),
        "P90": float(df['bert_input_len'].quantile(0.9)),
        "P95": float(df['bert_input_len'].quantile(0.95)),
        "P99": float(df['bert_input_len'].quantile(0.99)),
        "max": float(df['bert_input_len'].max()),
    },
    "overflow": overflow,
    "hist": hist_data,
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n✅ 统计已保存到 data/processed/final_data/maxlen_stats.json")
print("\n==== 给 jev / 人看的统计摘要 ====")
print(stats_text)

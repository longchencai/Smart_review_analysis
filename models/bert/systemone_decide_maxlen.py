# ============================================
# 用 jev / SystemOne 辅助决定 BERT 的 max_len
# 思路：max_len 本质是数据驱动的超参，百分位数能直接算出来；
#       这里用 jev 演示"把统计 + 权衡说明喂给模型，让它做受约束、可审计的选择"，
#       输出带理由，方便写进实验记录 / 给同学交代。
# 前置：pip install typesafe_sdk，且已设置 API key 环境变量（typesafe_sdk 读取的那个）
# 运行：conda activate nlp && python systemone_decide_maxlen.py
# ============================================
import json
import os
from typesafe_sdk import Choice, Noul, TypeSafeClient

# 项目根从本文件位置推算（本文件在 models/bert/ 下，向上三级即项目根）
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATS_JSON = os.path.join(BASE, "data", "processed", "final_data", "maxlen_stats.json")

# 读取之前 eda_maxlen_analysis.py 算好的统计（包含给模型看的 stats_text）
with open(STATS_JSON, "r", encoding="utf-8") as f:
    stats = json.load(f)
state_text = stats["stats_text"]

client = TypeSafeClient()

# ========= 问题定义 =========
questions_def = {
    # Choice：在候选 max_len 里选一个，附带每个选项的权衡
    "max_len_choice": Choice(
        instructions=(
            "Based on the review length statistics in `state`, choose the BEST max_len for a "
            "BERT Chinese sentiment+category classifier. Prefer the SMALLEST value that still "
            "retains at least 98% of reviews (i.e. truncation <= 2%). If 98% coverage needs a "
            "value that is clearly wasteful (e.g. far above P99), pick the next lower one. "
            "Explain the trade-off between coverage (information kept) and compute cost "
            "(padding waste, GPU memory, training speed)."
        ),
        criteria={
            "64": "Covers ~75% of data. Very fast/cheap, but truncates 24.7% of reviews (risky: long reviews often carry key sentiment).",
            "128": "Covers ~90% of data. Moderate cost, but still truncates ~10% of reviews.",
            "256": "Covers ~98% of data. Good balance: only ~2% truncated, modest padding overhead vs median length 36.",
            "512": "Covers ~99.5% of data. Almost no truncation, but ~14x padding vs median (36) -> slow and memory-heavy on an 8GB GPU.",
        },
    ),
    # Noul：截断 <2% 是否可接受（演示 Noul 用法，输出 0~1 概率）
    "truncation_acceptable": Noul(
        instructions=(
            "Given the stats, is it acceptable to truncate <2% of the longest reviews "
 "(i.e. choose max_len=256) for the sake of training efficiency, assuming the truncated "
 "reviews are outliers (copy-paste product specs, spam)? Answer as probability the trade-off is acceptable."
        )
    ),
}

# ========= 调用 system_one =========
resp = client.system_one(state=state_text, questions=questions_def)

# ========= 解析输出 =========
print("==== jev 对 max_len 的决策 ====")
print(f"推荐 max_len = {resp.answers['max_len_choice'].choice}")
print(f"截断 <2% 可接受概率 = {resp.answers['truncation_acceptable'].noul:.4f}")
print("\n==== 完整返回对象 ====")
print(resp)

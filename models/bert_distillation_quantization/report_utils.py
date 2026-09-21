# ============================================================
# 评估与结果落盘工具
#
# 统一「在验证集上评估 + 把指标写进 eval_result.txt」的逻辑，
# 让 distill_train.py（纯蒸馏）和 quantize.py（蒸馏+量化）产出格式一致、可直接对比。
# ============================================================

import io
import os
import time

import torch

from bert_config import Config
from bert_model_eval_utils import model_eval

config = Config()

LINE = "-" * 80
HEAVY = "=" * 80


def evaluate(model, dataloader, device=None, verbose=False):
    """在给定 dataloader 上评估，返回结构化指标字典。"""
    (ca, cp, cr, cf), (sa, sp, sr, sf) = model_eval(dataloader, model, device=device, verbose=verbose)
    return {
        'cat': {'acc': ca, 'pre': cp, 'rec': cr, 'f1': cf},
        'sent': {'acc': sa, 'pre': sp, 'rec': sr, 'f1': sf},
        'avg_f1': (cf + sf) / 2,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model):
    """按 state_dict 的实盘序列化体积估算模型文件大小（含 int8 量化张量）。"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024 ** 2


def file_size_mb(path):
    return os.path.getsize(path) / 1024 ** 2 if path and os.path.exists(path) else None


def format_metrics_block(index, title, subtitle, metrics, device,
                         param_count=None, size_mb=None, model_path=None, extra=None):
    """把一次评估结果格式化成一段文本。"""
    out = []
    out.append(HEAVY)
    out.append(f"【{index}】{title}")
    out.append(f"      {subtitle}")
    out.append(LINE)
    if model_path:
        out.append(f"  模型文件   : {model_path}")
    if param_count is not None:
        out.append(f"  参数量     : {param_count:,}")
    if size_mb is not None:
        out.append(f"  模型体积   : {size_mb:.2f} MB")
    out.append(f"  评估设备   : {device}")
    out.append(LINE)
    c, s = metrics['cat'], metrics['sent']
    out.append(f"  商品大类   : 准确率={c['acc']:.4f}  精确率={c['pre']:.4f}  召回率={c['rec']:.4f}  F1={c['f1']:.4f}")
    out.append(f"  情感二分类 : 准确率={s['acc']:.4f}  精确率={s['pre']:.4f}  召回率={s['rec']:.4f}  F1={s['f1']:.4f}")
    out.append(LINE)
    out.append(f"  综合 F1    : {metrics['avg_f1']:.4f}   (= 两个任务 macro-F1 的平均)")
    if extra:
        for line in extra:
            out.append(f"  {line}")
    out.append("")
    return "\n".join(out)


def format_compare_block(label_a, metrics_a, label_b, metrics_b):
    """对比两段指标的逐项差值（b - a）。"""
    out = [HEAVY, "【对比】量化带来的影响", LINE]
    out.append(f"  基准 : {label_a}")
    out.append(f"  对照 : {label_b}")
    out.append(LINE)
    out.append(f"  {'指标':<22}{'基准':>10}{'对照':>10}{'差值':>12}")
    for task_name, key in (("商品大类", "cat"), ("情感二分类", "sent")):
        for m_name, m_key in (("准确率", "acc"), ("精确率", "pre"), ("召回率", "rec"), ("macro-F1", "f1")):
            a = metrics_a[key][m_key]
            b = metrics_b[key][m_key]
            out.append(f"  {task_name + ' ' + m_name:<22}{a:>10.4f}{b:>10.4f}{b - a:>+12.4f}")
        out.append(LINE)
    a, b = metrics_a['avg_f1'], metrics_b['avg_f1']
    out.append(f"  {'综合 F1':<22}{a:>10.4f}{b:>10.4f}{b - a:>+12.4f}")
    out.append("")
    return "\n".join(out)


def reset_result_file():
    """清空结果文件。

    蒸馏训练是流水线的第一步，重跑时必须清掉上一次的旧结果，
    否则 eval_result.txt 会累积多份互相矛盾的历史段落。
    quantize.py 作为第二步只负责「追加」。
    """
    if os.path.exists(config.eval_result_path):
        os.remove(config.eval_result_path)
        print(f"已清空旧结果文件: {config.eval_result_path}")


def ensure_result_header():
    """首次写入时建文件头；已存在则不动。"""
    if os.path.exists(config.eval_result_path):
        return
    # 行数从 CSV 实际读，不写死：行数统计口径（按行 vs 按记录）容易差几条
    try:
        import pandas as pd
        n_val = len(pd.read_csv(config.val_path, encoding=config.csv_encoding))
    except Exception:
        n_val = '未知'
    header = [
        HEAVY,
        " BERT 蒸馏 (SmallBERT 4L/384) + 动态量化   验证集评估结果",
        HEAVY,
        f" 评估数据集 : data/processed/final_data/val.csv  ({n_val} 条)",
        f" 任务       : 商品大类 7 分类 + 情感二分类（双任务）",
        f" 指标口径   : 全部为 macro 平均（类别不平衡，accuracy 会误导）",
        f" 生成脚本   : cache_teacher_logits.py / distill_train.py / quantize.py",
        "",
        " 说明：所有条目均在同一份验证集、同一套评估代码下产生，可直接横向对比。",
        "",
    ]
    with open(config.eval_result_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(header) + "\n")


def append_block(text):
    ensure_result_header()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(config.eval_result_path, 'a', encoding='utf-8') as f:
        f.write(f"\n[写入时间 {stamp}]\n")
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    print(f"\n已写入评估结果: {config.eval_result_path}")

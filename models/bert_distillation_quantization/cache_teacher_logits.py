# ============================================================
# 步骤 1：离线预计算教师软标签
#
# 教师冻结，对 train.csv 全量前向一次，把「原始 logits」存成 float16 的 npz。
#
# 关键设计：
#   · 只存原始 logits，不预先除以温度 T —— 这样之后扫 T 不必重跑教师
#   · 按 CSV 原始行号顺序存储 —— dataloader 可以放心 shuffle，对齐由下标保证
#   · 存 float16 足够：BERT logits 量级通常在 ±15 内
#   · 跑完会顺手用缓存算一遍教师的「训练集准确率」，
#     若与模型实际水平明显不符，说明缓存与 CSV 错位了
#
# 运行环境：GPU（dl-gpu）
#   conda run -p C:\Users\29011\.conda\envs\dl-gpu python cache_teacher_logits.py
# ============================================================

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from bert_config import Config
from bert_classifier_model import MyBertMultiTaskClassifier

config = Config()


def main():
    if os.path.exists(config.teacher_soft_label_path) and '--force' not in sys.argv:
        print(f"软标签缓存已存在：{config.teacher_soft_label_path}")
        print("如需重建请加 --force")
        return

    print("=" * 68)
    print("步骤 1：预计算教师软标签")
    print("=" * 68)

    df = pd.read_csv(config.train_path, encoding=config.csv_encoding)
    texts = [str(t) for t in df['review']]
    y_cat = np.array([config.cat2id[str(v)] for v in df['cat_l1']], dtype=np.int64)
    y_sent = np.array([int(v) for v in df['label']], dtype=np.int64)
    n = len(texts)
    print(f"训练样本 : {n} 条   (max_len = {config.distill_max_len})")

    # 教师：加载权重、冻结、评估模式
    teacher = MyBertMultiTaskClassifier()
    state = torch.load(config.teacher_model_path, map_location='cpu', weights_only=True)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"教师权重与结构不匹配！missing={list(missing)} unexpected={list(unexpected)}")
    teacher.eval().to(config.device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"教师路径 : {config.teacher_model_path}")
    print(f"设备     : {config.device}")

    cat_out = np.zeros((n, config.cat_class_num), dtype=np.float16)
    sent_out = np.zeros((n, config.sent_class_num), dtype=np.float16)

    bs = config.cache_batch_size
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            enc = config.bert_tokenizer(
                texts[start:end],
                max_length=config.distill_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            enc = {k: v.to(config.device) for k, v in enc.items()}
            cat_logits, sent_logits = teacher(enc)
            cat_out[start:end] = cat_logits.float().cpu().numpy().astype(np.float16)
            sent_out[start:end] = sent_logits.float().cpu().numpy().astype(np.float16)

            if (start // bs) % 20 == 0 or end == n:
                done = end
                elapsed = time.time() - t0
                eta = elapsed / done * (n - done) if done else 0
                print(f"  {done:>6}/{n}  {done / n * 100:5.1f}%  已用 {elapsed:6.1f}s  预计剩余 {eta:6.1f}s")

    os.makedirs(os.path.dirname(config.teacher_soft_label_path), exist_ok=True)
    np.savez_compressed(config.teacher_soft_label_path, cat_logits=cat_out, sent_logits=sent_out)
    size_mb = os.path.getsize(config.teacher_soft_label_path) / 1024 ** 2
    print(f"\n已保存   : {config.teacher_soft_label_path}  ({size_mb:.2f} MB)")
    print(f"cat_logits: {cat_out.shape} {cat_out.dtype}")
    print(f"sent_logits: {sent_out.shape} {sent_out.dtype}")

    # ---- 对齐自检：用缓存 logits 反推教师训练集准确率 ----
    pred_cat = cat_out.astype(np.float32).argmax(axis=1)
    pred_sent = sent_out.astype(np.float32).argmax(axis=1)
    acc_cat = (pred_cat == y_cat).mean()
    acc_sent = (pred_sent == y_sent).mean()
    print("\n---- 对齐自检（用缓存 logits 反推教师训练集表现）----")
    print(f"  教师训练集 大类 acc = {acc_cat:.4f}")
    print(f"  教师训练集 情感 acc = {acc_sent:.4f}")
    print("  （训练集上教师应接近满分；若明显偏低，说明缓存与 CSV 错位）")
    print("\n步骤 1 完成")


if __name__ == '__main__':
    main()

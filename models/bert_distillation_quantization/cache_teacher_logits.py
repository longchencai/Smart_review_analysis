# ============================================================
# 步骤 1：离线预计算教师的两类监督信号
#
#   ① 输出层软标签  (N, 7) + (N, 2)  —— 输出层蒸馏用
#   ② [CLS] 隐状态  (N, 768) x 4 层  —— 逐层 hidden-state 蒸馏用
#
# 关键设计：
#   · 只存「原始 logits」，不预先除以温度 T —— 这样扫 T 不必重跑教师
#   · 隐状态只对齐 [CLS] 位置：整条序列对齐需要缓存
#       43820 x 256 x 768 x 4 层 ≈ 69 GB，而 [CLS] 只需约 270 MB。
#     对分类任务而言 [CLS] 正是最终被分类头读的那个位置。
#   · 全部按 CSV 原始行号顺序存储 —— dataloader 可以放心 shuffle，对齐由下标保证
#   · float16 足够：BERT logits 量级在 ±15 内，隐状态是 LayerNorm 后的 O(1) 量级
#   · 跑完会用缓存反推教师的「训练集准确率」，若明显偏低说明缓存与 CSV 错位
#
# 运行环境：GPU（dl-gpu）
#   conda run -p C:\Users\29011\.conda\envs\dl-gpu python cache_teacher_logits.py [--force]
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
    force = '--force' in sys.argv
    done = (os.path.exists(config.teacher_soft_label_path)
            and os.path.exists(config.teacher_hidden_cache_path))
    if done and not force:
        print(f"缓存已存在：\n  {config.teacher_soft_label_path}\n  {config.teacher_hidden_cache_path}")
        print("如需重建请加 --force")
        return

    print("=" * 70)
    print("步骤 1：预计算教师监督信号（软标签 + [CLS] 隐状态）")
    print("=" * 70)

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

    layer_ids = list(config.teacher_hidden_layer_map)
    teacher_hidden_dim = config.bert_config.hidden_size
    print(f"教师路径 : {config.teacher_model_path}")
    print(f"设备     : {config.device}")
    print(f"对齐层   : 学生第 1..{len(layer_ids)} 层  ←  教师第 {layer_ids} 层（hidden={teacher_hidden_dim}）")

    cat_out = np.zeros((n, config.cat_class_num), dtype=np.float16)
    sent_out = np.zeros((n, config.sent_class_num), dtype=np.float16)
    hid_out = np.zeros((len(layer_ids), n, teacher_hidden_dim), dtype=np.float16)

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
            # 直接调教师内部 bert 以拿到 hidden_states；后续两层复现其 forward 的全部计算
            out = teacher.bert(**enc, output_hidden_states=True)
            pooled = out['pooler_output']
            cat_logits = teacher.cat_linear(pooled)
            sent_logits = teacher.sent_linear(pooled)

            cat_out[start:end] = cat_logits.float().cpu().numpy().astype(np.float16)
            sent_out[start:end] = sent_logits.float().cpu().numpy().astype(np.float16)
            for k, li in enumerate(layer_ids):
                # [CLS] 是序列第 0 个位置；hidden_states[0] 是 embedding 输出，故层号即索引
                cls_h = out.hidden_states[li][:, 0, :]
                hid_out[k, start:end] = cls_h.float().cpu().numpy().astype(np.float16)

            if (start // bs) % 20 == 0 or end == n:
                elapsed = time.time() - t0
                eta = elapsed / end * (n - end) if end else 0
                print(f"  {end:>6}/{n}  {end / n * 100:5.1f}%  已用 {elapsed:6.1f}s  预计剩余 {eta:6.1f}s")

    os.makedirs(os.path.dirname(config.teacher_soft_label_path), exist_ok=True)
    np.savez_compressed(config.teacher_soft_label_path, cat_logits=cat_out, sent_logits=sent_out)
    np.savez_compressed(config.teacher_hidden_cache_path, cls_hidden=hid_out)

    print(f"\n软标签  -> {config.teacher_soft_label_path}  "
          f"({os.path.getsize(config.teacher_soft_label_path) / 1024 ** 2:.2f} MB)  "
          f"cat{cat_out.shape} sent{sent_out.shape}")
    print(f"隐状态  -> {config.teacher_hidden_cache_path}  "
          f"({os.path.getsize(config.teacher_hidden_cache_path) / 1024 ** 2:.2f} MB)  "
          f"cls_hidden{hid_out.shape}")

    # ---- 对齐自检 ----
    pred_cat = cat_out.astype(np.float32).argmax(axis=1)
    pred_sent = sent_out.astype(np.float32).argmax(axis=1)
    print("\n---- 对齐自检（用缓存的 logits 反推教师训练集表现）----")
    print(f"  教师训练集 大类 acc = {(pred_cat == y_cat).mean():.4f}")
    print(f"  教师训练集 情感 acc = {(pred_sent == y_sent).mean():.4f}")
    print(f"  隐状态形状 {hid_out.shape}  范数均值 {np.linalg.norm(hid_out.astype(np.float32), axis=2).mean():.2f}")
    print("  （训练集上教师应接近满分；若明显偏低说明缓存与 CSV 错位）")
    print("\n步骤 1 完成")


if __name__ == '__main__':
    main()

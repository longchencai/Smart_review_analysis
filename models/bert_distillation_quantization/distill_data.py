# ============================================================
# 蒸馏专用数据管道
#
# 每条样本除「文本 + 情感标签 + 大类标签」外，可带两类教师监督信号：
#   ① 输出层软 logits          (7,) + (2,)          —— 输出层蒸馏
#   ② 4 个对齐层的 [CLS] 隐状态 (4, 768)             —— 逐层 hidden-state 蒸馏（实验 B 才启用）
#
# 为什么要离线缓存：
#   教师冻结，对 43820 条只需前向一次。缓存之后：
#     1. 训练时不装教师，显存只装学生
#     2. 每个 epoch 不再重复前向教师（教师前向 8.3 min/epoch，比学生训练本身还慢）
#     3. 扫 T / alpha 不必重跑教师
#   两类信号都按「原始 CSV 行号」存取，所以 dataloader 可以放心 shuffle。
# ============================================================

import numpy as np
import pandas as pd
import torch

from bert_config import Config

config = Config()


class DistillDataset(torch.utils.data.Dataset):
    """返回 (文本, 情感标签, 大类标签, cat软标签, sent软标签[, 教师CLS隐状态])"""

    def __init__(self, csv_path, cache_path, hidden_cache_path=None):
        df = pd.read_csv(csv_path, encoding=config.csv_encoding)
        self.texts = [str(t) for t in df['review']]
        self.sent_labels = [int(v) for v in df['label']]
        self.cat_labels = [config.cat2id[str(v)] for v in df['cat_l1']]

        cache = np.load(cache_path)
        self.cat_soft = cache['cat_logits']      # (N, 7)  float16
        self.sent_soft = cache['sent_logits']    # (N, 2)  float16

        n = len(self.texts)
        if not (n == len(self.cat_soft) == len(self.sent_soft)):
            raise ValueError(
                f"软标签缓存与 CSV 行数不一致：CSV {n} 条，"
                f"cat_logits {len(self.cat_soft)} 条，sent_logits {len(self.sent_soft)} 条。\n"
                f"请确认 {cache_path} 是用同一份 train.csv 生成的（重新运行 cache_teacher_logits.py --force）。"
            )
        if self.cat_soft.shape[1] != config.cat_class_num:
            raise ValueError(f"缓存的 cat_logits 维度 {self.cat_soft.shape[1]} != {config.cat_class_num}")
        if self.sent_soft.shape[1] != config.sent_class_num:
            raise ValueError(f"缓存的 sent_logits 维度 {self.sent_soft.shape[1]} != {config.sent_class_num}")

        # ---- 可选的逐层 [CLS] 隐状态 ----
        self.cls_hidden = None          # (L, N, 768) float16
        if hidden_cache_path:
            if not config.distill_use_hidden_loss:
                raise ValueError("传入了 hidden_cache_path，但 config.distill_use_hidden_loss 为 False")
            data = np.load(hidden_cache_path)
            self.cls_hidden = data['cls_hidden']
            L, n_h, d = self.cls_hidden.shape
            if n_h != n:
                raise ValueError(
                    f"隐状态缓存条数 {n_h} 与 CSV {n} 不一致，请重新运行 cache_teacher_logits.py --force")
            if L != len(config.teacher_hidden_layer_map):
                raise ValueError(f"隐状态层数 {L} != teacher_hidden_layer_map 长度 "
                                 f"{len(config.teacher_hidden_layer_map)}")
            if d != config.bert_config.hidden_size:
                raise ValueError(f"隐状态维度 {d} != 教师 hidden_size {config.bert_config.hidden_size}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        base = (self.texts[index], self.sent_labels[index], self.cat_labels[index],
                self.cat_soft[index], self.sent_soft[index])
        if self.cls_hidden is None:
            return base
        return base + (self.cls_hidden[:, index, :],)     # (L, 768)


def _encode(texts):
    return config.bert_tokenizer(
        list(texts),
        max_length=config.distill_max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )


def distill_collate_fn(batch):
    """不带隐状态：teacher 软标签 + 硬标签"""
    texts, sent_labels, cat_labels, cat_soft, sent_soft = zip(*batch)
    return (
        _encode(texts),
        torch.tensor(sent_labels, dtype=torch.long),
        torch.tensor(cat_labels, dtype=torch.long),
        torch.tensor(np.stack(cat_soft), dtype=torch.float32),
        torch.tensor(np.stack(sent_soft), dtype=torch.float32),
    )


def distill_hidden_collate_fn(batch):
    """带隐状态：额外返回 (B, L, 768) 的教师 [CLS] 隐状态"""
    texts, sent_labels, cat_labels, cat_soft, sent_soft, cls_hidden = zip(*batch)
    return (
        _encode(texts),
        torch.tensor(sent_labels, dtype=torch.long),
        torch.tensor(cat_labels, dtype=torch.long),
        torch.tensor(np.stack(cat_soft), dtype=torch.float32),
        torch.tensor(np.stack(sent_soft), dtype=torch.float32),
        # np.stack -> (B, L, 768)；float16 -> float32
        torch.tensor(np.stack(cls_hidden), dtype=torch.float32),
    )


def build_distill_dataloader(shuffle=True, batch_size=None):
    """按 config.distill_use_hidden_loss 自动选择是否加载隐状态缓存。"""
    use_hidden = config.distill_use_hidden_loss
    dataset = DistillDataset(
        config.train_path,
        config.teacher_soft_label_path,
        config.teacher_hidden_cache_path if use_hidden else None,
    )
    bs = batch_size or config.distill_batch_size
    collate = distill_hidden_collate_fn if use_hidden else distill_collate_fn
    return torch.utils.data.DataLoader(
        dataset, bs, shuffle=shuffle, collate_fn=collate, num_workers=0
    )


if __name__ == '__main__':
    dl = build_distill_dataloader()
    print(f"蒸馏 dataloader: {len(dl.dataset)} 条, 每批 {config.distill_batch_size}, 共 {len(dl)} 批")
    print(f"是否启用逐层隐状态: {config.distill_use_hidden_loss}")
    batch = next(iter(dl))
    print("特征 shape   :", {k: tuple(v.shape) for k, v in batch[0].items()})
    print("情感硬标签   :", tuple(batch[1].shape))
    print("大类硬标签   :", tuple(batch[2].shape))
    print("cat 软标签   :", tuple(batch[3].shape), batch[3].dtype)
    print("sent 软标签  :", tuple(batch[4].shape), batch[4].dtype)
    if len(batch) > 5:
        print("教师CLS隐状态:", tuple(batch[5].shape), batch[5].dtype)

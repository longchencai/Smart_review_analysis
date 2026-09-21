# ============================================================
# 蒸馏专用数据管道
#
# 与 dataloader_utils.py 的区别：这里每条样本除了「文本 + 情感标签 + 大类标签」，
# 还带一份「教师对该样本输出的 soft logits」（cat 7 维 / sent 2 维）。
#
# 为什么要离线缓存软标签：
#   教师是冻结的，对训练集 43834 条只需前向一次。缓存之后：
#     1. 训练时显存只装学生，不必同时装教师
#     2. 每个 epoch 不再重复前向教师，训练快很多
#     3. 扫 T / alpha 时不必重跑教师
#   注意软标签按「原始 CSV 行号」存取，因此 dataloader 可以放心 shuffle。
# ============================================================

import numpy as np
import pandas as pd
import torch

from bert_config import Config

config = Config()


class DistillDataset(torch.utils.data.Dataset):
    """(文本, 情感标签, 大类标签, 教师cat logits, 教师sent logits)"""

    def __init__(self, csv_path, cache_path):
        df = pd.read_csv(csv_path, encoding=config.csv_encoding)
        self.texts = [str(t) for t in df['review']]
        self.sent_labels = [int(v) for v in df['label']]
        self.cat_labels = [config.cat2id[str(v)] for v in df['cat_l1']]

        cache = np.load(cache_path)
        self.cat_soft = cache['cat_logits']      # (N, 7)  float16
        self.sent_soft = cache['sent_logits']    # (N, 2)  float16

        if not (len(self.texts) == len(self.cat_soft) == len(self.sent_soft)):
            raise ValueError(
                f"软标签缓存与 CSV 行数不一致：CSV {len(self.texts)} 条，"
                f"cat_logits {len(self.cat_soft)} 条，sent_logits {len(self.sent_soft)} 条。\n"
                f"请确认 {cache_path} 是用同一份 train.csv 生成的（重新运行 cache_teacher_logits.py）。"
            )
        if self.cat_soft.shape[1] != config.cat_class_num:
            raise ValueError(f"缓存的 cat_logits 维度 {self.cat_soft.shape[1]} != {config.cat_class_num}")
        if self.sent_soft.shape[1] != config.sent_class_num:
            raise ValueError(f"缓存的 sent_logits 维度 {self.sent_soft.shape[1]} != {config.sent_class_num}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (self.texts[index],
                self.sent_labels[index],
                self.cat_labels[index],
                self.cat_soft[index],
                self.sent_soft[index])


def distill_collate_fn(batch):
    """
    :param batch: [(文本, 情感标签, 大类标签, cat软标签, sent软标签), ...]
    :return: (tokenizer特征dict, 情感硬标签, 大类硬标签, cat软标签, sent软标签)
    """
    texts, sent_labels, cat_labels, cat_soft, sent_soft = zip(*batch)

    enc = config.bert_tokenizer(
        list(texts),
        max_length=config.distill_max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    return (
        enc,
        torch.tensor(sent_labels, dtype=torch.long),
        torch.tensor(cat_labels, dtype=torch.long),
        torch.tensor(np.stack(cat_soft), dtype=torch.float32),    # float16 -> float32
        torch.tensor(np.stack(sent_soft), dtype=torch.float32),
    )


def build_distill_dataloader(shuffle=True, batch_size=None):
    dataset = DistillDataset(config.train_path, config.teacher_soft_label_path)
    bs = batch_size or config.distill_batch_size
    return torch.utils.data.DataLoader(
        dataset, bs, shuffle=shuffle, collate_fn=distill_collate_fn, num_workers=0
    )


if __name__ == '__main__':
    dl = build_distill_dataloader()
    print(f"蒸馏 dataloader: {len(dl.dataset)} 条, 每批 {config.distill_batch_size}, 共 {len(dl)} 批")
    enc, sent_y, cat_y, cat_soft, sent_soft = next(iter(dl))
    print("特征 shape   :", {k: tuple(v.shape) for k, v in enc.items()})
    print("情感硬标签   :", tuple(sent_y.shape), sent_y[:8].tolist())
    print("大类硬标签   :", tuple(cat_y.shape), cat_y[:8].tolist())
    print("cat 软标签   :", tuple(cat_soft.shape), cat_soft.dtype)
    print("sent 软标签  :", tuple(sent_soft.shape), sent_soft.dtype)
    print("软标签第 0 条:", cat_soft[0].tolist())

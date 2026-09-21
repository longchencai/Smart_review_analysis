# ============================================
# 生成 FastText 格式数据（双任务版本）
# 输出：商品大类任务 / 情感任务 的 train/val/test.txt
# 格式：__label__类别 文本
# ============================================

import pandas as pd
import os
import jieba

# ---------- 配置 ----------
DATA_DIR = "../data/processed/final_data/"
os.makedirs(DATA_DIR, exist_ok=True)

# 停用词
STOPWORDS_PATH = f"{DATA_DIR}/stopwords.txt"
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])


def preprocess_text(text):
    """jieba 分词 + 去停用词"""
    text = str(text).replace('\t', ' ').replace('\n', ' ')
    words = jieba.lcut(text)
    words = [w for w in words if w not in stopwords and w.strip()]
    return ' '.join(words)


def write_fasttext_format(df, label_col, label_map, output_path):
    """
    把 df 写成 fastText 格式
    :param df: 数据框
    :param label_col: 标签列名
    :param label_map: 标签值 -> 标签名字典
    :param output_path: 输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = preprocess_text(row['review'])
            label_name = f"__label__{label_map[row[label_col]]}"
            f.write(f"{label_name} {text}\n")


# 读数据
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

# 商品大类标签映射
l1_names = sorted(train_df['cat_l1'].unique())
cat_id2name = {i: name for i, name in enumerate(l1_names)}

# 情感标签映射
sent_id2name = {0: '负面评价', 1: '正面评价'}

print(f"商品大类类别：{cat_id2name}")
print(f"情感类别：{sent_id2name}")

# ---------- 生成商品大类任务数据 ----------
write_fasttext_format(train_df, 'cat_l1', cat_id2name, f"{DATA_DIR}/train_fasttext_cat.txt")
write_fasttext_format(val_df, 'cat_l1', cat_id2name, f"{DATA_DIR}/val_fasttext_cat.txt")
write_fasttext_format(test_df, 'cat_l1', cat_id2name, f"{DATA_DIR}/test_fasttext_cat.txt")
print(f"✅ 商品大类 fastText 数据已生成")

# ---------- 生成情感任务数据 ----------
write_fasttext_format(train_df, 'label', sent_id2name, f"{DATA_DIR}/train_fasttext_sent.txt")
write_fasttext_format(val_df, 'label', sent_id2name, f"{DATA_DIR}/val_fasttext_sent.txt")
write_fasttext_format(test_df, 'label', sent_id2name, f"{DATA_DIR}/test_fasttext_sent.txt")
print(f"✅ 情感 fastText 数据已生成")

# 验证一下
print(f"\n=== 验证商品大类 train.txt 前 5 行 ===")
with open(f"{DATA_DIR}/train_fasttext_cat.txt", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  {line.strip()[:80]}...")

print(f"\n=== 验证情感 train.txt 前 5 行 ===")
with open(f"{DATA_DIR}/train_fasttext_sent.txt", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"  {line.strip()[:80]}...")

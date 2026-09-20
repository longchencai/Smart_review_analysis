# ============================================
# 生成FastText格式数据
# 输出：class.txt, train.txt, val.txt, test.txt
# 格式：文本<Tab>类别编号
# ============================================

import pandas as pd
import os

# ---------- 配置 ----------
DATA_DIR = "../data/processed/final_data/"

# 读数据
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

# 读类别名（7个一级大类）
with open(f"{DATA_DIR}/class_names_l1.txt", 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

# 建立类别到编号的映射
class_to_id = {name: i for i, name in enumerate(class_names)}

print(f"类别列表：{class_names}")
print(f"类别映射：{class_to_id}")

# ---------- 写 class.txt ----------
with open(f"{DATA_DIR}/class.txt", 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(name + '\n')
print(f"\n✅ class.txt 已生成（{len(class_names)}个类别）")

# ---------- 写 train.txt / val.txt / test.txt ----------
def write_fasttext_format(df, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = str(row['review']).replace('\t', ' ').replace('\n', ' ')
            label_id = class_to_id[row['cat_l1']]
            f.write(f"{text}\t{label_id}\n")

write_fasttext_format(train_df, f"{DATA_DIR}/train.txt")
write_fasttext_format(val_df, f"{DATA_DIR}/val.txt")
write_fasttext_format(test_df, f"{DATA_DIR}/test.txt")

print(f"✅ train.txt 已生成（{len(train_df)}条）")
print(f"✅ val.txt 已生成（{len(val_df)}条）")
print(f"✅ test.txt 已生成（{len(test_df)}条）")

# 验证一下
print(f"\n=== 验证 train.txt 前5行 ===")
with open(f"{DATA_DIR}/train.txt", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        text, label = line.strip().rsplit('\t', 1)
        print(f"  [{label}] {text[:50]}...")

# ============================================
# 生成FastText格式数据
# 输出：class.txt（标签编号定义）, train.txt, val.txt, test.txt
# 格式：文本<Tab>类别编号
# ============================================

import csv
import os

import pandas as pd

# ---------- 配置 ----------
# 基于脚本自身位置解析，因此在任何目录下运行都可以
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data", "processed", "final_data"))


def data_file(filename):
    """在 final_data 目录内拼出文件路径，并校验没有越出该目录。"""
    path = os.path.abspath(os.path.join(DATA_DIR, filename))
    if os.path.dirname(path) != DATA_DIR:
        raise ValueError(f"路径越界，已拒绝：{filename}")
    return path


# 读数据（数据 CSV 带 BOM，需显式指定 encoding）
train_df = pd.read_csv(data_file("train.csv"), encoding='utf-8-sig')
val_df = pd.read_csv(data_file("val.csv"), encoding='utf-8-sig')
test_df = pd.read_csv(data_file("test.csv"), encoding='utf-8-sig')

# 类别名直接从数据推导（sorted 顺序），不依赖额外的名称文件
class_names = sorted(set(train_df['cat_l1']) | set(val_df['cat_l1']) | set(test_df['cat_l1']))

# 建立类别到编号的映射
class_to_id = {name: i for i, name in enumerate(class_names)}

print(f"类别列表：{class_names}")
print(f"类别映射：{class_to_id}")

# ---------- 写 class.txt ----------
# 用 to_csv 而不是 open() 写文件，路径已由 data_file() 校验在数据目录内
pd.Series(class_names, name='cat_l1').to_csv(
    data_file("class.txt"), index=False, header=False, encoding='utf-8'
)
print(f"\n✅ class.txt 已生成（{len(class_names)}个类别）")


# ---------- 写 train.txt / val.txt / test.txt ----------
def write_fasttext_format(frame, output_name):
    out = pd.DataFrame({
        'review': frame['review'].astype(str).str.replace(r'[\t\r\n]+', ' ', regex=True),
        'label_id': frame['cat_l1'].map(class_to_id),
    })
    if out['label_id'].isnull().any():
        raise ValueError("有类别不在 class_names 中，无法映射为编号")
    # QUOTE_NONE：保持「文本<Tab>编号」的裸格式，不给字段加引号（文本中的制表符/换行已替换掉）
    out.to_csv(data_file(output_name), sep='\t', index=False, header=False,
               encoding='utf-8', quoting=csv.QUOTE_NONE)


write_fasttext_format(train_df, "train.txt")
write_fasttext_format(val_df, "val.txt")
write_fasttext_format(test_df, "test.txt")

print(f"✅ train.txt 已生成（{len(train_df)}条）")
print(f"✅ val.txt 已生成（{len(val_df)}条）")
print(f"✅ test.txt 已生成（{len(test_df)}条）")

# 验证一下
print("\n=== 验证 train.txt 前5行 ===")
with open(data_file("train.txt"), 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        text, label = line.strip().rsplit('\t', 1)
        print(f"  [{label}] {text[:50]}...")

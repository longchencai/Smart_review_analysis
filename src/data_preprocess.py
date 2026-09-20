# ============================================
# 电商评论9分类 - 数据清洗脚本
# 功能：读入数据 → 清洗 → 划分数据集 → 保存
# ============================================

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# ---------- 【配置路径】 ----------
RAW_DATA_PATH = "../data/raw/online_shopping_10_cats.csv"
PROCESSED_DIR = "../data/processed/"

# 保存到 exp_b_9class 目录，不覆盖之前的10类数据
SAVE_DIR = f"{PROCESSED_DIR}/exp_b_9class/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 【第1步：读入原始数据】 ----------
print("=" * 50)
print("第1步：读入原始数据...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"原始数据行数：{len(df)}")
print(f"原始数据列名：{df.columns.tolist()}")

# ---------- 【第2步：删除样本太少的类别】 ----------
print("\n" + "=" * 50)
print("第2步：删除样本太少的类别（热水器）...")
print(f"删除前类别数：{df['cat'].nunique()}")

df = df[df['cat'] != '热水器']  # 热水器只有575条，删掉

print(f"删除后类别数：{df['cat'].nunique()}")
print(f"删除后剩余数据：{len(df)} 行")

print("\n【各类别数量分布】：")
print(df['cat'].value_counts())

# ---------- 【第3步：数据清洗】 ----------
print("\n" + "=" * 50)
print("第3步：开始数据清洗...")

# 3.1 去重
print("3.1 去除重复行...")
df = df.drop_duplicates()
print(f"去重后剩余：{len(df)} 行")

# 3.2 去除空值
print("3.2 去除空评论...")
df = df.dropna(subset=['review'])
print(f"去空后剩余：{len(df)} 行")

# 3.3 去除超短评论（少于5个字的）
print("3.3 去除超短评论（少于5个字）...")
df = df[df['review'].str.len() >= 5]
print(f"去短后剩余：{len(df)} 行")

# 3.4 清洗文本内容
print("3.4 清洗文本中的特殊符号...")

def clean_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['review'] = df['review'].apply(clean_text)

df = df[df['review'].str.len() >= 5]
print(f"文本清洗后剩余：{len(df)} 行")

# ---------- 【第4步：数据集划分】 ----------
print("\n" + "=" * 50)
print("第4步：划分数据集...")

train_df, temp_df = train_test_split(
    df, 
    test_size=0.3,
    random_state=42,
    stratify=df['cat']
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5,
    random_state=42,
    stratify=temp_df['cat']
)

print(f"训练集：{len(train_df)} 条")
print(f"验证集：{len(val_df)} 条")
print(f"测试集：{len(test_df)} 条")

# ---------- 【第5步：保存数据】 ----------
print("\n" + "=" * 50)
print("第5步：保存处理好的数据...")

train_df.to_csv(f"{SAVE_DIR}/train.csv", index=False, encoding='utf-8-sig')
val_df.to_csv(f"{SAVE_DIR}/val.csv", index=False, encoding='utf-8-sig')
test_df.to_csv(f"{SAVE_DIR}/test.csv", index=False, encoding='utf-8-sig')

class_names = sorted(df['cat'].unique())
with open(f"{SAVE_DIR}/class_names.txt", 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(name + '\n')

print(f"\n✅ 全部完成！数据已保存到 {SAVE_DIR}")
print(f"   - train.csv: {len(train_df)} 条")
print(f"   - val.csv: {len(val_df)} 条")
print(f"   - test.csv: {len(test_df)} 条")
print(f"   - class_names.txt: {len(class_names)} 个类别")
print(f"   类别列表：{class_names}")

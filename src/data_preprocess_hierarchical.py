# ============================================
# 电商评论分类 - 数据清洗脚本（两级分类体系版 v2）
# 功能：读入数据 → 类别重命名 → 构建两级分类 → 清洗 → 划分数据集 → 保存
#
# 一级大类（7个）：数码电子、个护美妆、食品饮料、服饰鞋包、图书文娱、本地生活、家用电器
# 二级小类（10个）：手机、平板、电脑、洗发水、乳制品、水果、衣服、书籍、酒店、热水器
# ============================================

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# ---------- 【配置路径】 ----------
RAW_DATA_PATH = "../data/raw/online_shopping_10_cats.csv"
PROCESSED_DIR = "../data/processed/"

# 保存到 final_data 目录（最终数据）
SAVE_DIR = f"{PROCESSED_DIR}/final_data/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 【类别映射配置】 ----------
# 原始类别名 → 二级小类（规范命名）
CAT_RENAME = {
    '计算机': '电脑',
    '蒙牛': '乳制品',
    # 其他类别名保持不变
    '书籍': '书籍',
    '平板': '平板',
    '手机': '手机',
    '水果': '水果',
    '洗发水': '洗发水',
    '衣服': '衣服',
    '酒店': '酒店',
    '热水器': '热水器',
}

# 二级小类 → 一级大类
L2_TO_L1 = {
    '手机': '数码电子',
    '平板': '数码电子',
    '电脑': '数码电子',
    '洗发水': '个护美妆',
    '乳制品': '食品饮料',
    '水果': '食品饮料',
    '衣服': '服饰鞋包',
    '书籍': '图书文娱',
    '酒店': '本地生活',
    '热水器': '家用电器',
}

# ---------- 【第1步：读入原始数据】 ----------
print("=" * 50)
print("第1步：读入原始数据...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"原始数据行数：{len(df)}")
print(f"原始数据列名：{df.columns.tolist()}")

# ---------- 【第2步：构建两级分类体系】 ----------
print("\n" + "=" * 50)
print("第2步：构建两级分类体系...")

# 2.1 重命名二级小类
df['cat_l2'] = df['cat'].map(CAT_RENAME)

# 2.2 映射一级大类
df['cat_l1'] = df['cat_l2'].map(L2_TO_L1)

# 2.3 删除原来的cat列
df = df.drop(columns=['cat'])

# 检查有没有映射失败的
if df['cat_l1'].isnull().any() or df['cat_l2'].isnull().any():
    print("❌ 错误：有类别没有映射成功！")
    print(df[df['cat_l1'].isnull()]['cat_l2'].unique())
    exit(1)

print("\n【一级大类分布】：")
l1_dist = df['cat_l1'].value_counts()
print(l1_dist)
print(f"\n一级大类数量：{df['cat_l1'].nunique()}")

print("\n【二级小类分布】：")
l2_dist = df['cat_l2'].value_counts()
print(l2_dist)
print(f"\n二级小类数量：{df['cat_l2'].nunique()}")

print("\n【两级分类对应关系】：")
hierarchy = df.groupby(['cat_l1', 'cat_l2']).size().reset_index(name='数量')
for _, row in hierarchy.iterrows():
    print(f"  {row['cat_l1']} → {row['cat_l2']}：{row['数量']}条")

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
    text = re.sub(r'<.*?>', '', text)          # 去除HTML标签
    text = re.sub(r'http\S+', '', text)         # 去除URL
    text = re.sub(r'\s+', ' ', text).strip()    # 合并空白字符
    return text

df['review'] = df['review'].apply(clean_text)
df = df[df['review'].str.len() >= 5]
print(f"文本清洗后剩余：{len(df)} 行")

# ---------- 【第4步：数据集划分（按二级小类分层）】 ----------
print("\n" + "=" * 50)
print("第4步：划分数据集（按二级小类分层）...")

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['cat_l2']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['cat_l2']
)

print(f"训练集：{len(train_df)} 条")
print(f"验证集：{len(val_df)} 条")
print(f"测试集：{len(test_df)} 条")

# ---------- 【第5步：保存数据】 ----------
print("\n" + "=" * 50)
print("第5步：保存处理好的数据...")

# 调整列顺序：review, label, cat_l1, cat_l2
cols = ['review', 'label', 'cat_l1', 'cat_l2']
train_df = train_df[cols]
val_df = val_df[cols]
test_df = test_df[cols]

train_df.to_csv(f"{SAVE_DIR}/train.csv", index=False, encoding='utf-8-sig')
val_df.to_csv(f"{SAVE_DIR}/val.csv", index=False, encoding='utf-8-sig')
test_df.to_csv(f"{SAVE_DIR}/test.csv", index=False, encoding='utf-8-sig')

# 保存一级大类名称
l1_names = sorted(df['cat_l1'].unique())
with open(f"{SAVE_DIR}/class_names_l1.txt", 'w', encoding='utf-8') as f:
    for name in l1_names:
        f.write(name + '\n')

# 保存二级小类名称
l2_names = sorted(df['cat_l2'].unique())
with open(f"{SAVE_DIR}/class_names_l2.txt", 'w', encoding='utf-8') as f:
    for name in l2_names:
        f.write(name + '\n')

# 保存两级分类映射表
hierarchy.to_csv(f"{SAVE_DIR}/category_hierarchy.csv", index=False, encoding='utf-8-sig')

print(f"\n✅ 全部完成！数据已保存到 {SAVE_DIR}")
print(f"   - train.csv: {len(train_df)} 条")
print(f"   - val.csv: {len(val_df)} 条")
print(f"   - test.csv: {len(test_df)} 条")
print(f"   - class_names_l1.txt: {len(l1_names)} 个一级大类 → {l1_names}")
print(f"   - class_names_l2.txt: {len(l2_names)} 个二级小类 → {l2_names}")
print(f"   - category_hierarchy.csv: 两级分类映射表")

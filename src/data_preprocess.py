# ============================================
# 电商评论分类 - 数据清洗脚本
# 功能：读入数据 → 映射一级大类 → 清洗（含 PII 脱敏）→ 去重 → 划分数据集 → 保存
#
# 任务为 7 个一级大类分类（另配情感二分类），因此输出不含二级小类列。
# 原始 10 个类别 → 7 个大类的对应关系：
#   数码电子 ← 手机、平板、计算机
#   个护美妆 ← 洗发水
#   食品饮料 ← 水果、蒙牛
#   服饰鞋包 ← 衣服
#   图书文娱 ← 书籍
#   本地生活 ← 酒店
#   家用电器 ← 热水器
# ============================================

import html
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- 【配置路径】 ----------
RAW_DATA_PATH = "../data/raw/online_shopping_10_cats.csv"
PROCESSED_DIR = "../data/processed/"

# 保存到 final_data 目录（最终数据）
SAVE_DIR = f"{PROCESSED_DIR}/final_data/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 【类别映射配置】 ----------
# 原始类别名 → 一级大类
CAT_TO_L1 = {
    '手机': '数码电子',
    '平板': '数码电子',
    '计算机': '数码电子',
    '洗发水': '个护美妆',
    '水果': '食品饮料',
    '蒙牛': '食品饮料',
    '衣服': '服饰鞋包',
    '书籍': '图书文娱',
    '酒店': '本地生活',
    '热水器': '家用电器',
}

# ---------- 【文本清洗规则】 ----------
# PII 只遮盖有强特征（11 位手机号、带区号座机）或带明确标签（电话/QQ/订单号…）的数字，
# 不做「长数字一律遮盖」，否则会误伤商品型号、价格等正常内容。
MOBILE_RE = re.compile(r'1[3-9]\d{9}')
LANDLINE_RE = re.compile(r'0\d{2,3}-\d{7,8}')
LABELED_ID_RE = re.compile(
    r'(电话|手机号|手机|QQ|qq|Qq|微信|VX|vx|订单号|订单号码|单号)\s*[:：]?\s*\d[\d\-]{4,}(?![A-Za-z])'
)
EMAIL_RE = re.compile(r'[\w.\-+]+@[\w\-]+(?:\.[\w\-]+)+')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_RE = re.compile(r'<.*?>')
PII_MARK_RE = re.compile(r'\[手机号\]|\[电话\]|\[邮箱\]|\[已脱敏\]')
# 判定「有实义内容」：至少含一个中文、字母或数字
MEANINGFUL_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')


def mask_pii(text):
    """遮盖个人信息。用占位符替换而不是删掉整句，保留原句结构。"""
    text = EMAIL_RE.sub('[邮箱]', text)
    text = MOBILE_RE.sub('[手机号]', text)
    text = LANDLINE_RE.sub('[电话]', text)
    text = LABELED_ID_RE.sub(lambda m: f"{m.group(1)}[已脱敏]", text)
    return text


def clean_text(text):
    text = str(text)
    text = HTML_TAG_RE.sub('', text)      # 去 HTML 标签
    text = html.unescape(text)            # 去 HTML 实体（&#183; &hellip; &quot; 等）
    text = URL_RE.sub('', text)           # 去 URL（含无协议头的裸 www.）
    text = re.sub(r'\s+', ' ', text).strip()
    return mask_pii(text)


# ---------- 【第1步：读入原始数据】 ----------
print("=" * 50)
print("第1步：读入原始数据...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"原始数据行数：{len(df)}")
print(f"原始数据列名：{df.columns.tolist()}")

# ---------- 【第2步：映射一级大类】 ----------
print("\n" + "=" * 50)
print("第2步：映射一级大类...")

df['cat_l1'] = df['cat'].map(CAT_TO_L1)

# 检查有没有映射失败的（在删掉原 cat 列之前检查，方便定位是哪个原始类别）
if df['cat_l1'].isnull().any():
    print("❌ 错误：有类别没有映射成功！")
    print(df.loc[df['cat_l1'].isnull(), 'cat'].unique())
    exit(1)

df = df.drop(columns=['cat'])

print("\n【一级大类分布】：")
print(df['cat_l1'].value_counts())
print(f"\n一级大类数量：{df['cat_l1'].nunique()}")

# ---------- 【第3步：数据清洗】 ----------
print("\n" + "=" * 50)
print("第3步：开始数据清洗...")

# 3.1 去除空评论
print("3.1 去除空评论...")
n0 = len(df)
df = df.dropna(subset=['review'])
print(f"    剩余 {len(df)} 行（去掉 {n0 - len(df)}）")

# 3.2 去除超短评论（少于5个字）
print("3.2 去除超短评论（少于5个字）...")
n0 = len(df)
df = df[df['review'].str.len() >= 5]
print(f"    剩余 {len(df)} 行（去掉 {n0 - len(df)}）")

# 3.3 清洗文本：HTML 标签 / HTML 实体 / URL / 空白，并做 PII 脱敏
print("3.3 清洗文本（HTML、URL、空白）并脱敏个人信息...")
df['review'] = df['review'].apply(clean_text)
n0 = len(df)
df = df[df['review'].str.len() >= 5]
print(f"    剩余 {len(df)} 行（清洗后长度不足5字去掉 {n0 - len(df)}）")
n_pii = df['review'].str.contains(PII_MARK_RE, regex=True, na=False).sum()
print(f"    其中 {n_pii} 行的手机号/邮箱/订单号等已被遮盖为占位符")

# 3.4 去除纯符号噪声行
# 注意：不能用「不含中文就删」——无中文的行里有合法英文评论（如 waste of my money...），
# 删掉会丢真实数据。这里只删「连一个中文/字母/数字都没有」的纯符号行。
print("3.4 去除纯符号噪声行...")
n0 = len(df)
df = df[df['review'].str.contains(MEANINGFUL_RE, regex=True, na=False)]
print(f"    剩余 {len(df)} 行（去掉 {n0 - len(df)}）")

# 3.5 去重：同一（一级大类, 评论）只保留一条，label 取多数票
# 必须放在文本清洗之后，才能把仅空白差异的重复合并；同时避免同一文本分别落进
# train 和 test 造成泄漏。原脚本用 df.drop_duplicates() 对全列去重，实际一行都删不掉。
print("3.5 去除重复评论（同一大类内，label 取多数票）...")
n0 = len(df)
majority = df.groupby(['cat_l1', 'review'])['label'].transform(
    lambda s: s.value_counts().idxmax()
)
df = df.assign(label=majority).drop_duplicates(subset=['cat_l1', 'review'], keep='first')
print(f"    剩余 {len(df)} 行（去掉 {n0 - len(df)}）")

print("\n【清洗后一级大类分布】：")
print(df['cat_l1'].value_counts())

# ---------- 【第4步：数据集划分（按一级大类 + 情感组合分层）】 ----------
print("\n" + "=" * 50)
print("第4步：划分数据集（按一级大类 + 情感组合分层）...")

# 只按类别分层会让正负样本在三份集合之间漂移，导致 val 与 test 的情感基线
# 不可比（val 上调出的结论无法迁移到 test）。因此把 label 一并放进分层键。
def strat_key(frame):
    return frame['cat_l1'].astype(str) + '_' + frame['label'].astype(str)

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=strat_key(df)
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=strat_key(temp_df)
)

print(f"训练集：{len(train_df)} 条")
print(f"验证集：{len(val_df)} 条")
print(f"测试集：{len(test_df)} 条")

# ---------- 【第5步：保存数据】 ----------
print("\n" + "=" * 50)
print("第5步：保存处理好的数据...")

# 调整列顺序：review, label, cat_l1
cols = ['review', 'label', 'cat_l1']
train_df = train_df[cols]
val_df = val_df[cols]
test_df = test_df[cols]

train_df.to_csv(f"{SAVE_DIR}/train.csv", index=False, encoding='utf-8-sig')
val_df.to_csv(f"{SAVE_DIR}/val.csv", index=False, encoding='utf-8-sig')
test_df.to_csv(f"{SAVE_DIR}/test.csv", index=False, encoding='utf-8-sig')

l1_names = sorted(df['cat_l1'].unique())

print(f"\n✅ 全部完成！数据已保存到 {SAVE_DIR}")
print(f"   - train.csv: {len(train_df)} 条")
print(f"   - val.csv: {len(val_df)} 条")
print(f"   - test.csv: {len(test_df)} 条")
print(f"   - 一级大类（{len(l1_names)} 个）：{l1_names}")
print(f"   - 标签编号文件 class.txt 与 FastText 格式数据由 generate_fasttext_data.py 生成")

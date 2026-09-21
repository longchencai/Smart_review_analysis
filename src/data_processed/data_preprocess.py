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
# 基于脚本自身位置解析，因此在任何目录下运行都可以
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "online_shopping_10_cats.csv")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "final_data")

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

# ---------- 【清洗规则】 ----------
# 说明：本数据集里尖括号内容基本都是中文书名（<阿波林小世界>、<<十万个为什么>>），
# 不是 HTML 标签，所以只用「以字母开头的真标签」模式，否则会把书名整段吃掉。
# 第二段分支处理残缺的闭标签片段（如抓取残留的 </Feature），"</" 不是正常中文内容。
HTML_TAG_RE = re.compile(r'</?[A-Za-z][^<>]*>|</[A-Za-z][^<>]*')
# 邮箱只认 ASCII。若用 \w 会匹配中文，导致「不是很好@裤腿太小.码数也不对」被整句
# 误判为邮箱并替换掉，进而被「不足5字」过滤删除。
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
URL_RE = re.compile(r'https?:\S+|www\.\S+')

# PII 遮盖：只动「有强特征」或「带明确标签」的数字/账号，避免误伤商品型号、价格、日期。
# 统一加数字边界 (?<!\d) / (?!\d)，防止在长数字中间截断出假号码。
MOBILE_RE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')                    # 11 位手机号
# 其余电话类都替换成同一个占位符，合并为一条减少重复调用：
# 带连字符座机 / 不带连字符座机 / 400、800 服务热线
PHONE_RE = re.compile(
    r'(?<!\d)0\d{2,3}-\d{7,8}(?!\d)'
    r'|(?<!\d)0\d{9,11}(?!\d)'
    r'|(?<![\d-])[48]00[-\s]?\d{3}[-\s]?\d{4}(?!\d)'
)
# 带标签的号码：数字串后面不能紧跟字母或数字，避免把「手机mate7」「手机10000mAh」当号码。
# 标签与号码之间允许空格/冒号，也允许「是、为」这类连接词（如「订单号是1401482205」）；
# 裸「订单」放在最后，用于「订单184477629」这种不带「号」字的写法。
LABELED_DIGIT_RE = re.compile(
    r'(?P<label>电话|手机号|手机|QQ|qq|Qq|qQ|微信|VX|vx|Vx|vX'
    r'|订单号码|订单编号|订单号|发票号码|发票号|运单号|快递单号|物流单号|会员号|投诉编号|投诉单号|单号|订单)'
    r'[\s:：是为]{0,3}(?P<id>\d[\d\-]{4,})(?![\dA-Za-z])'
)
# 号码写在标签之前的写法（如「174208675是我的订单编号」）。
# 这里不收录裸「订单」——反向写法里它和「订单数量／金额」歧义，正向规则里才无歧义。
REVERSED_ID_RE = re.compile(
    r'(?P<id>(?<!\d)\d{6,}(?!\d))\s*(?:是|为)\s*我?的?\s*'
    r'(?P<label>订单编号|订单号码|订单号|发票号码|发票号|运单号|快递单号|物流单号|会员号|投诉编号|投诉单号|单号)'
)
# 字母数字型账号只对微信/QQ 生效（VX 会与笔记本型号 vx600x 冲突，故不纳入）
ALNUM_ID_RE = re.compile(
    r'(?P<label>微信|QQ|qq|Qq|qQ)'
    r'[\s:：是为]{0,3}(?=[A-Za-z0-9\-_]*\d)(?P<id>[A-Za-z0-9\-_]{4,})'
)
PII_MARK_RE = re.compile(r'\[手机号\]|\[电话\]|\[邮箱\]|\[已脱敏\]')
# 判定「有实义内容」：至少含一个中文、字母或数字
MEANINGFUL_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')


def mask_pii(text):
    """遮盖个人信息。用占位符替换而不是删掉整句，保留原句结构。"""
    text = EMAIL_RE.sub('[邮箱]', text)
    text = MOBILE_RE.sub('[手机号]', text)
    text = PHONE_RE.sub('[电话]', text)
    text = LABELED_DIGIT_RE.sub(lambda m: f"{m.group('label')}[已脱敏]", text)
    text = REVERSED_ID_RE.sub(lambda m: f"{m.group('label')}[已脱敏]", text)
    text = ALNUM_ID_RE.sub(lambda m: f"{m.group('label')}[已脱敏]", text)
    return text


def clean_text(text):
    text = str(text)
    # 先去 HTML 实体再去标签：实体解码后若产生真标签可被清除，
    # 而「价格&lt;100元&gt;」这类不含字母的尖括号内容不会被误删。
    text = html.unescape(text)
    text = HTML_TAG_RE.sub('', text)
    text = URL_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return mask_pii(text)


# ---------- 【第1步：读入原始数据】 ----------
print("=" * 50)
print("第1步：读入原始数据...")

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"找不到原始数据：{RAW_DATA_PATH}")

df = pd.read_csv(RAW_DATA_PATH)
n_raw = len(df)
print(f"原始数据行数：{n_raw}")
print(f"原始数据列名：{df.columns.tolist()}")

# ---------- 【第2步：映射一级大类】 ----------
print("\n" + "=" * 50)
print("第2步：映射一级大类...")

for col in ('cat', 'label', 'review'):
    if col not in df.columns:
        raise KeyError(f"原始数据缺少列：{col}")

df['cat_l1'] = df['cat'].map(CAT_TO_L1)

# 检查有没有映射失败的（在删掉原 cat 列之前检查，方便定位是哪个原始类别）
if df['cat_l1'].isnull().any():
    bad = df.loc[df['cat_l1'].isnull(), 'cat'].unique().tolist()
    raise ValueError(f"以下原始类别没有映射规则：{bad}")

df = df.drop(columns=['cat'])

print("\n【一级大类分布】：")
print(df['cat_l1'].value_counts())
print(f"\n一级大类数量：{df['cat_l1'].nunique()}")

# ---------- 【第3步：数据清洗】 ----------
print("\n" + "=" * 50)
print("第3步：开始数据清洗...")

# 3.1 去除空评论
n0 = len(df)
df = df.dropna(subset=['review'])
n_empty = n0 - len(df)

# 3.2 清洗文本：HTML 实体/标签、URL、空白，并做 PII 脱敏
df['review'] = df['review'].apply(clean_text)

# 3.3 去除过短评论（少于5个字）
# 只在这里过滤一次即可：清洗只会让文本变短，因此无需在清洗前再过滤一遍。
n0 = len(df)
df = df[df['review'].str.len() >= 5]
n_short = n0 - len(df)

# 3.4 去除纯符号噪声行
# 注意：不能用「不含中文就删」——无中文的行里有合法英文评论（如 waste of my money...），
# 删掉会丢真实数据。这里只删「连一个中文/字母/数字都没有」的纯符号行。
n0 = len(df)
df = df[df['review'].str.contains(MEANINGFUL_RE, regex=True, na=False)]
n_symbol = n0 - len(df)

# 3.5 去重：同一（一级大类, 评论）只保留一条
# 不做「多数票」：实测 18 个重复组中有 17 组恰好是两条且 label 相反，从来不存在多数票，
# 所以明确采用「保留先出现的一条」(keep='first') 作为平票规则。
n0 = len(df)
df = df.drop_duplicates(subset=['cat_l1', 'review'], keep='first')
n_dup = n0 - len(df)

# PII 计数放在去重之后，保证与最终交付的数据一致
n_pii = df['review'].str.contains(PII_MARK_RE, regex=True, na=False).sum()

print(f"3.1 空评论     ：去 {n_empty}")
print("3.2 文本清洗   ：HTML 实体/标签、URL、空白 + PII 脱敏")
print(f"3.3 不足5字    ：去 {n_short}")
print(f"3.4 纯符号行   ：去 {n_symbol}")
print(f"3.5 重复评论   ：去 {n_dup}")
print(f"    —— 合计去 {n_raw - len(df)} 行，剩余 {len(df)} 行")
print(f"    —— 其中 {n_pii} 行的手机号/邮箱/订单号等已遮盖为占位符")

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

os.makedirs(SAVE_DIR, exist_ok=True)

# 调整列顺序：review, label, cat_l1
cols = ['review', 'label', 'cat_l1']
train_df = train_df[cols]
val_df = val_df[cols]
test_df = test_df[cols]

train_df.to_csv(os.path.join(SAVE_DIR, "train.csv"), index=False, encoding='utf-8-sig')
val_df.to_csv(os.path.join(SAVE_DIR, "val.csv"), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(SAVE_DIR, "test.csv"), index=False, encoding='utf-8-sig')

l1_names = sorted(df['cat_l1'].unique())

print(f"\n✅ 全部完成！数据已保存到 {SAVE_DIR}")
print(f"   - train.csv: {len(train_df)} 条")
print(f"   - val.csv: {len(val_df)} 条")
print(f"   - test.csv: {len(test_df)} 条")
print(f"   - 一级大类（{len(l1_names)} 个）：{l1_names}")
print("   - 标签编号文件 class.txt 与 FastText 格式数据由 generate_fasttext_data.py 生成")

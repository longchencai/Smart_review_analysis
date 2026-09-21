# ============================================
# 数据质量审计脚本（只读，不写任何文件）
#
# 打印 README「已知局限与注意事项」一节引用的各项指标，便于一键复核。
# 用法：python src/data_audit.py
# ============================================

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data", "processed", "final_data"))
RAW_PATH = os.path.abspath(os.path.join(BASE_DIR, "data", "raw", "online_shopping_10_cats.csv"))

SPLITS = ('train', 'val', 'test')
frames = {s: pd.read_csv(os.path.join(DATA_DIR, f"{s}.csv"), encoding='utf-8-sig') for s in SPLITS}
df = pd.concat(frames.values(), ignore_index=True)
n = len(df)

print("=" * 62)
print("数据质量审计报告")
print("=" * 62)

# ---------- 1. 规模与划分 ----------
print("\n【1】规模与划分")
for s in SPLITS:
    print(f"  {s:<6} {len(frames[s]):>6} 条  ({len(frames[s]) / n * 100:.2f}%)")
print(f"  合计   {n:>6} 条")

# ---------- 2. 类别分布与正负比例 ----------
print("\n【2】类别分布与正负比例（label=1 占比）")
print(f"  {'类别':<8}{'样本':>7}{'占比':>8}{'正面率':>9}   划分漂移(pp)")
max_drift = 0.0
for c in df['cat_l1'].value_counts().index:
    sub = df[df['cat_l1'] == c]
    rates = [frames[s].loc[frames[s]['cat_l1'] == c, 'label'].mean() * 100 for s in SPLITS]
    drift = max(rates) - min(rates)
    max_drift = max(max_drift, drift)
    print(f"  {c:<8}{len(sub):>7}{len(sub) / n * 100:>7.2f}%{sub['label'].mean() * 100:>8.2f}%   {drift:>6.2f}")
print(f"  → 各类最大漂移 {max_drift:.2f}pp（应 < 0.5pp）")

# ---------- 3. 关键词基线（不做任何训练） ----------
KW = {
    '数码电子': ['手机', '平板', '电脑', '计算机', '屏幕', '分辨率'],
    '食品饮料': ['水果', '苹果', '蒙牛', '牛奶', '酸奶', '好吃'],
    '个护美妆': ['洗发水', '洗发', '头发', '头屑'],
    '服饰鞋包': ['衣服', '尺码', '面料', '穿着', '裤子'],
    '本地生活': ['酒店', '房间', '入住', '前台', '退房'],
    '图书文娱': ['书', '作者', '内容', '这本'],
    '家用电器': ['热水器', '热水', '洗澡', '燃气'],
}
POS = ['好评', '值得购买', '满意', '喜欢', '不错', '很好', '推荐', '正品', '实惠', '划算',
       '惊喜', '完美', '给力', '棒', '好用', '舒服']
NEG = ['差评', '垃圾', '失望', '差劲', '太差', '不好', '退货', '上当', '坑', '骗',
       '劣质', '后悔', '投诉', '假货', '难用', '太烂', '无语']


def argmax_l1(text):
    best, best_score = None, -1
    for cls, kws in KW.items():
        score = sum(text.count(w) for w in kws)
        if score > best_score:
            best, best_score = cls, score
    return best if best_score > 0 else None


pred_l1 = df['review'].apply(argmax_l1)
acc_l1 = (pred_l1 == df['cat_l1']).mean() * 100


def sentiment_rule(text):
    p = sum(text.count(w) for w in POS)
    q = sum(text.count(w) for w in NEG)
    return None if p == q else int(p > q)


pred_sent = df['review'].apply(sentiment_rule)
covered = pred_sent.notna()
acc_covered = (pred_sent[covered] == df.loc[covered, 'label']).mean() * 100
# 未命中关键词的行按「负面」计入（保守做法）；这不是模型指标，只是规则基线
acc_all = (pred_sent.fillna(0).astype(int) == df['label']).mean() * 100

print("\n【3】关键词基线（无需训练）")
print(f"  {'一级大类':<8}{'关键词覆盖率':>12}")
for c in KW:
    sub = df[df['cat_l1'] == c]
    hit = sub['review'].str.contains('|'.join(KW[c]), regex=True, na=False).mean() * 100
    print(f"  {c:<8}{hit:>11.1f}%")
print(f"  → 关键词 argmax 基线准确率：{acc_l1:.1f}%")
print(f"  → 情感关键词规则：覆盖率 {covered.mean() * 100:.1f}%，覆盖部分准确率 {acc_covered:.1f}%，"
      f"整体准确率 {acc_all:.2f}%")

# ---------- 4. 蒙牛子类的关键词依赖（需原始数据） ----------
print("\n【4】蒙牛子类（现并入食品饮料）的关键词依赖")
if os.path.exists(RAW_PATH):
    raw = pd.read_csv(RAW_PATH).dropna(subset=['review'])
    mn = raw.loc[raw['cat'] == '蒙牛', 'review']
    print(f"  原始 {len(mn)} 条，其中字面含「蒙牛」的占 {mn.str.contains('蒙牛').mean() * 100:.1f}%")
else:
    print(f"  跳过：找不到原始数据 {RAW_PATH}")

# ---------- 5. 长度分布 ----------
lengths = df['review'].str.len()
print("\n【5】评论长度（字符）")
print(f"  最小 {lengths.min()}  中位 {lengths.median():.0f}  平均 {lengths.mean():.0f}  "
      f"P95 {lengths.quantile(0.95):.0f}  P99 {lengths.quantile(0.99):.0f}  最大 {lengths.max()}")
print(f"  >128 字：{(lengths > 128).mean() * 100:.1f}%    >510 字：{(lengths > 510).mean() * 100:.2f}%")
for c in ('图书文娱', '本地生活'):
    print(f"  {c} 中位数字数：{df.loc[df['cat_l1'] == c, 'review'].str.len().median():.0f}")

# ---------- 6. 重复与跨类歧义 ----------
print("\n【6】重复与跨类歧义")


def norm_text(frame):
    """归一化评论文本（合并空白、去首尾），用于判定「同一文本」。"""
    return frame['review'].str.replace(r'\s+', ' ', regex=True).str.strip()


norm = norm_text(df)
tagged = df.assign(_key=norm)
dup_rows = len(df) - norm.nunique()
per_class_dup = len(df) - len(df.drop_duplicates(subset=['cat_l1', 'review']))
cross = tagged.groupby('_key')['cat_l1'].nunique()
cross_texts = int((cross > 1).sum())
cross_rows = int(tagged['_key'].isin(cross[cross > 1].index).sum())
keyed = {s: norm_text(frames[s]) for s in SPLITS}
leak_test = int(keyed['test'].isin(set(keyed['train'])).sum())
leak_val = int(keyed['val'].isin(set(keyed['train'])).sum())
print(f"  完全重复的文本行：{dup_rows}（同一大类内重复 {per_class_dup} 行）")
print(f"  同一文本出现在多个大类：{cross_texts} 条文本 / {cross_rows} 行"
      f"（占 {cross_rows / n * 100:.2f}%）")
print(f"  在 train 中有完全相同文本的：test {leak_test} 行、val {leak_val} 行")
both_label = tagged.groupby('_key')['label'].nunique()
print(f"  同一文本带两种情感标签：{int((both_label > 1).sum())} 条")

# ---------- 7. 残留噪声自检 ----------
print("\n【7】残留噪声自检（应全为 0）")
checks = {
    'HTML 标签': r'</?[A-Za-z][^<>]*>',
    'HTML 实体': r'&[a-zA-Z]+;|&#\d+;',
    'URL': r'https?://\S+|www\.\S+',
    '邮箱': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    '11位手机号': r'(?<!\d)1[3-9]\d{9}(?!\d)',
    '座机(含连字符)': r'(?<!\d)0\d{2,3}-\d{7,8}(?!\d)',
    '座机(无连字符)': r'(?<!\d)0\d{9,11}(?!\d)',
    'Tab/换行': r'[\t\r\n]',
    '连续空格': r'  ',
}
for name, pattern in checks.items():
    hit = df['review'].str.contains(pattern, regex=True, na=False).sum()
    print(f"  {name:<14}: {hit} 行 {'OK' if hit == 0 else '<-- 需检查'}")
print(f"  {'空评论':<14}: {df['review'].isna().sum()} 行")
print(f"  {'不足5字':<14}: {(lengths < 5).sum()} 行")
print(f"  PII 占位符     : {df['review'].str.contains(r'\[手机号\]|\[电话\]|\[邮箱\]|\[已脱敏\]', regex=True, na=False).sum()} 行（已遮盖）")
print("\n审计完成（本脚本不修改任何文件）")

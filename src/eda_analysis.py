# ============================================
# EDA探索性数据分析 - 优化版（9张图）
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from wordcloud import WordCloud
import os

# ---------- 配置 ----------
DATA_PATH = "../data/processed/final_data/train.csv"
FIG_DIR = "../data/figures/"
STOPWORDS_PATH = "../data/processed/final_data/stopwords.txt"

os.makedirs(FIG_DIR, exist_ok=True)

# 统一风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

# 配色统一
COLOR_POS = '#52C41A'
COLOR_NEG = '#EA6668'
COLOR_PRIMARY = '#9BBBF4'
COLOR_ORANGE = '#F4B393'

df = pd.read_csv(DATA_PATH)
print(f"读入数据：{len(df)} 条")

# 读停用词
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])

# ---------- 先做基础统计 ----------
print("\n=== 基础数据统计 ===")
print(f"总样本数：{len(df)}")
print(f"缺失值：\n{df.isnull().sum()}")
print(f"重复评论数：{df.duplicated().sum()} ({df.duplicated().sum()/len(df)*100:.1f}%)")

df['review_length'] = df['review'].str.len()
p99 = df['review_length'].quantile(0.99)
print(f"评论长度：平均{df['review_length'].mean():.0f}字，P99={p99:.0f}字")

# ============================================
# 图1：标签分布（正负情感）
# ============================================
print("\n图1：标签分布...")
plt.figure(figsize=(7, 5))
sentiment = df['label'].value_counts()
colors = [COLOR_NEG if i == 0 else COLOR_POS for i in sentiment.index]
bars = plt.bar(['负面评价', '正面评价'], sentiment.values, color=colors, width=0.6)
plt.title(f'正负情感标签分布（共{len(df)}条）', fontsize=13)
plt.ylabel('评论数量', fontsize=11)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_label_distribution.png", bbox_inches='tight')
plt.close()

# ============================================
# 图2：类别分布
# ============================================
print("图2：类别分布...")
plt.figure(figsize=(9, 6))
cat_counts = df['cat_l1'].value_counts()
bars = plt.barh(cat_counts.index, cat_counts.values, color=COLOR_PRIMARY)
plt.title(f'商品类别分布（共{df["cat_l1"].nunique()}个类别）', fontsize=13)
plt.xlabel('评论数量', fontsize=11)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 50, bar.get_y() + bar.get_height()/2.,
             f'{int(width)}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_category_distribution.png", bbox_inches='tight')
plt.close()

# ============================================
# 图3：类别×标签 热力图（带百分比）
# ============================================
print("图3：类别×标签热力图...")
plt.figure(figsize=(9, 6))
heatmap_data = pd.crosstab(df['cat_l1'], df['label'], normalize='index') * 100
heatmap_data.columns = ['负面%', '正面%']
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', cbar_kws={'label': '占比 (%)'})
plt.title('各类别下的正负情感比例', fontsize=13)
plt.xlabel('情感', fontsize=11)
plt.ylabel('商品类别', fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_category_label_heatmap.png", bbox_inches='tight')
plt.close()

# ============================================
# 图4：评论长度分布（P99截断）
# ============================================
print("图4：评论长度分布...")
plt.figure(figsize=(9, 5))
plt.hist(df[df['review_length'] <= p99]['review_length'], bins=50, 
         color='#A2DDAA', edgecolor='white')
plt.axvline(df['review_length'].mean(), color='red', linestyle='--', 
            label=f'平均：{df["review_length"].mean():.0f}字')
plt.title(f'评论长度分布（显示至P99={p99:.0f}字）', fontsize=13)
plt.xlabel('评论字数', fontsize=11)
plt.ylabel('评论数量', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_review_length_distribution.png", bbox_inches='tight')
plt.close()

# ============================================
# 图5：按情感分的评论长度箱线图（P99截断）
# ============================================
print("图5：按情感分的评论长度...")
plt.figure(figsize=(7, 6))
sns.boxplot(x='label', y='review_length', data=df, palette=[COLOR_NEG, COLOR_POS])
plt.ylim(0, p99)
plt.title(f'不同情感的评论长度对比（P99={p99:.0f}）', fontsize=13)
plt.xlabel('情感（0=负面, 1=正面）', fontsize=11)
plt.ylabel('评论字数', fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_review_length_by_label.png", bbox_inches='tight')
plt.close()

# ============================================
# 图6：按类别分的评论长度
# ============================================
print("图6：按类别分的评论长度...")
plt.figure(figsize=(9, 6))
cat_length = df.groupby('cat_l1')['review_length'].mean().sort_values()
bars = plt.barh(cat_length.index, cat_length.values, color=COLOR_ORANGE)
plt.title('各类别平均评论长度', fontsize=13)
plt.xlabel('平均字数', fontsize=11)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
             f'{width:.0f}字', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_review_length_by_category.png", bbox_inches='tight')
plt.close()

# ============================================
# 图7：高频重复评论TOP10（带占比）
# ============================================
print("图7：高频重复评论TOP10...")
top_reviews = df['review'].value_counts().head(10)
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_reviews)), top_reviews.values, color='#E1B98F')
plt.yticks(range(len(top_reviews)), 
           [f'{r[:25]}...' if len(r)>25 else r for r in top_reviews.index])
plt.title(f'高频重复评论TOP10（重复率{df.duplicated().sum()/len(df)*100:.1f}%）', fontsize=13)
plt.xlabel('出现次数', fontsize=11)
for i, (v, bar) in enumerate(zip(top_reviews.values, bars)):
    plt.text(v + 1, i, f'{v}次 ({v/len(df)*100:.2f}%)', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_top_repeated_reviews.png", bbox_inches='tight')
plt.close()

# ============================================
# 图8：正负评论高频词对比
# ============================================
print("图8：正负评论高频词对比...")

def get_top_words(texts, n=20):
    words_list = []
    for text in texts:
        words = jieba.lcut(str(text))
        words = [w for w in words if w not in stopwords and len(w) > 1]
        words_list.extend(words)
    return pd.Series(words_list).value_counts().head(n)

positive_words = get_top_words(df[df['label'] == 1]['review'])
negative_words = get_top_words(df[df['label'] == 0]['review'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

ax1.barh(range(len(positive_words)), positive_words.values, color=COLOR_POS)
ax1.set_yticks(range(len(positive_words)))
ax1.set_yticklabels(positive_words.index)
ax1.set_title('正面评论高频词TOP20', fontsize=12)
ax1.set_xlabel('出现次数')

ax2.barh(range(len(negative_words)), negative_words.values, color=COLOR_NEG)
ax2.set_yticks(range(len(negative_words)))
ax2.set_yticklabels(negative_words.index)
ax2.set_title('负面评论高频词TOP20', fontsize=12)
ax2.set_xlabel('出现次数')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_top_words_by_label.png", bbox_inches='tight')
plt.close()

# ============================================
# 图9：词云图（新增）
# ============================================
print("图9：词云图...")

all_words = []
for text in df['review']:
    words = jieba.lcut(str(text))
    words = [w for w in words if w not in stopwords and len(w) > 1]
    all_words.extend(words)

text = ' '.join(all_words)

wc = WordCloud(
    font_path='simhei.ttf',
    width=800,
    height=600,
    background_color='white',
    stopwords=stopwords,
    max_words=100
)
wc.generate(text)

plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('评论高频词词云', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_wordcloud.png", bbox_inches='tight')
plt.close()

print(f"\n✅ 全部9张图完成！保存在：{FIG_DIR}")



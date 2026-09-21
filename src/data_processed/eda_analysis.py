# ============================================
# EDA探索性数据分析 - 优化版（9张图）
# ============================================

import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import jieba
from wordcloud import WordCloud

# ---------- 配置 ----------
# 基于脚本自身位置解析，因此在任何目录下运行都可以
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "final_data")

DATA_PATH = os.path.join(DATA_DIR, "train.csv")
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")
FIG_DIR = os.path.join(BASE_DIR, "data", "figures", "final_data")  # 与数据版本同名，避免和旧数据集混在一起

os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 中文字体（跨平台探测） ----------
# 各操作系统的中文字体名不同，仓库里也不带字体文件。找不到时 matplotlib 只是回退
# 默认字体（中文显示成方框），但 WordCloud 会直接抛 OSError —— 所以先用它探测一次，
# 由结果决定是否跳过图 9，而不是让脚本崩在已经写完图 1-8 之后。
FONT_CANDIDATES = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Hiragino Sans GB',
    'Noto Sans CJK SC', 'Source Han Sans CN', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
]


def pick_font_file():
    """返回本机第一个可用中文字体的文件路径；一个都没有则返回 None。"""
    for name in FONT_CANDIDATES:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name),
                                         fallback_to_default=False)
        except ValueError:
            continue
        if os.path.exists(path):
            return path
    return None


FONT_FILE = pick_font_file()

# 统一风格
if FONT_FILE is None:
    print("⚠️ 未检测到中文字体：图中中文可能显示为方框，图 9（词云）将被跳过。")
    print(f"   修复办法：安装任一中文字体，或把字体名/路径加入 FONT_CANDIDATES：{FONT_CANDIDATES}")
else:
    plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=FONT_FILE).get_name(),
                                       'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

# 配色统一
COLOR_POS = '#52C41A'
COLOR_NEG = '#EA6668'
COLOR_PRIMARY = '#9BBBF4'
COLOR_ORANGE = '#F4B393'

df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')   # 数据 CSV 带 BOM，需显式指定
print(f"读入数据：{len(df)} 条")

# 读停用词
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f.readlines()])

# ---------- 先做基础统计 ----------
print("\n=== 基础数据统计 ===")
print(f"总样本数：{len(df)}")
print(f"缺失值：\n{df.isnull().sum()}")
n_dup = df.duplicated().sum()
print(f"重复评论数：{n_dup} ({n_dup/len(df)*100:.1f}%)")

df['review_length'] = df['review'].str.len()
p99 = df['review_length'].quantile(0.99)
print(f"评论长度：平均{df['review_length'].mean():.0f}字，P99={p99:.0f}字")

# ============================================
# 图1：标签分布（正负情感）
# ============================================
print("\n图1：标签分布...")
plt.figure(figsize=(7, 5))
# 必须按 label 取值 (0, 1) 显式取数。value_counts() 是按数量降序返回的，直接接它的
# values 会在 label=1 数量更多时把「负面/正面」两个柱子的数值和颜色对调。
counts = df['label'].value_counts().reindex([0, 1], fill_value=0)
bars = plt.bar(['负面评价', '正面评价'], counts.values,
               color=[COLOR_NEG, COLOR_POS], width=0.6)
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
mean_len = df['review_length'].mean()
plt.hist(df[df['review_length'] <= p99]['review_length'], bins=50, 
         color='#A2DDAA', edgecolor='white')
plt.axvline(mean_len, color='red', linestyle='--', label=f'平均：{mean_len:.0f}字')
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
sns.boxplot(x='label', y='review_length', data=df, hue='label',
            palette=[COLOR_NEG, COLOR_POS], legend=False)
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
vc = df['review'].value_counts()
top_reviews = vc.head(10)
# 清洗阶段已按（大类, 文本）去重，所以不存在整行重复；这里统计的是「文本重复」的规模，
# 避免标题用整行重复率（恒为 0）而与图中出现 2 次的柱子自相矛盾。
dup_row_share = vc[vc >= 2].sum() / len(df) * 100
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_reviews)), top_reviews.values, color='#E1B98F')
plt.yticks(range(len(top_reviews)), 
           [f'{r[:25]}...' if len(r)>25 else r for r in top_reviews.index])
plt.title(f'高频重复评论文本TOP10（重复文本的评论占比{dup_row_share:.2f}%，同文本最多出现{vc.max()}次）', fontsize=12)
plt.xlabel('出现次数', fontsize=11)
plt.xlim(0, max(top_reviews.max() * 1.8, 3))   # 留出右侧空间，否则柱顶标注会被裁掉
for i, (v, bar) in enumerate(zip(top_reviews.values, bars)):
    plt.text(v + 0.05, i, f'{v}次', va='center', fontsize=9)
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

if FONT_FILE is None:
    print("   跳过：本机没有可用中文字体，WordCloud 无法渲染中文")
else:
    wc = WordCloud(
        font_path=FONT_FILE,
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

n_figs = 8 if FONT_FILE is None else 9
print(f"\n✅ 完成 {n_figs} 张图，保存在：{FIG_DIR}")
if FONT_FILE is None:
    print("   注意：若该目录里仍有 09_wordcloud.png，那是上一次运行留下的旧图")



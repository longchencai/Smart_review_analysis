import pandas as pd
import pickle
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 配置路径
DATA_PATH = "../data/processed/final_data/train.csv"
SAVE_DIR = "../models/rf"
os.makedirs(SAVE_DIR, exist_ok=True)
STOPWORDS_PATH = "../data/processed/final_data/stopwords.txt"

# 读取停用词
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f.readlines()])


def segment_text(text):
    """jieba 分词 + 去停用词"""
    words = jieba.lcut(str(text))
    words = [w for w in words if w not in stop_words and w.strip()]
    return ' '.join(words)


# 第一步：读取数据
df = pd.read_csv(DATA_PATH)
print(f"训练样本数: {len(df)}")
print(df.head(5))

# 对评论分词
print("正在进行 jieba 分词...")
df['words'] = df['review'].apply(segment_text)

# 商品大类标签：字符串 -> 整数
cat2id = {name: i for i, name in enumerate(sorted(df['cat_l1'].unique()))}
df['cat_label'] = df['cat_l1'].map(cat2id)

# 第二步：划分训练集和测试集
x_train_text, x_test_text, y_train, y_test = train_test_split(
    df['words'], df['cat_label'], test_size=0.2, random_state=22, stratify=df['cat_label']
)

# 第三步：将文本转换为数值特征
tfidf = TfidfVectorizer(max_features=10000)
x_train = tfidf.fit_transform(x_train_text)
x_test = tfidf.transform(x_test_text)

print(f"训练集特征 shape: {x_train.shape}")
print(f"测试集特征 shape: {x_test.shape}")

# 第四步：训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=22, n_jobs=-1)
print("训练商品大类随机森林模型...")
model.fit(x_train, y_train)

# 第五步：模型预测并评估
y_pred = model.predict(x_test)
print("\n商品大类随机森林评估结果:")
print("准确率:", accuracy_score(y_test, y_pred))
print("精确率 (macro):", precision_score(y_test, y_pred, average='macro', zero_division=0))
print("召回率 (macro):", recall_score(y_test, y_pred, average='macro', zero_division=0))
print("F1 分数 (macro):", f1_score(y_test, y_pred, average='macro', zero_division=0))

# 第六步：保存模型和向量化器
print("保存模型和向量化器...")
with open(f"{SAVE_DIR}/rf_cat_model.pkl", 'wb') as f:
    pickle.dump(model, f)
with open(f"{SAVE_DIR}/tfidf_cat_vectorizer.pkl", 'wb') as f:
    pickle.dump(tfidf, f)

# 保存类别映射
with open(f"{SAVE_DIR}/rf_cat_id2name.pkl", 'wb') as f:
    pickle.dump({v: k for k, v in cat2id.items()}, f)

print("商品大类随机森林模型保存成功！")

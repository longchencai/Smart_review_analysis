import jieba
import pickle
import warnings

warnings.filterwarnings('ignore')

# 1、加载模型和向量化器（模块被导入时加载一次，服务启动后所有请求共用）
SAVE_DIR = "../models/rf"

with open(f"{SAVE_DIR}/rf_cat_model.pkl", 'rb') as f:
    cat_model = pickle.load(f)
with open(f"{SAVE_DIR}/tfidf_cat_vectorizer.pkl", 'rb') as f:
    cat_tfidf = pickle.load(f)
with open(f"{SAVE_DIR}/rf_cat_id2name.pkl", 'rb') as f:
    cat_id2name = pickle.load(f)

with open(f"{SAVE_DIR}/rf_sent_model.pkl", 'rb') as f:
    sent_model = pickle.load(f)
with open(f"{SAVE_DIR}/tfidf_sent_vectorizer.pkl", 'rb') as f:
    sent_tfidf = pickle.load(f)

# 情感标签映射
sent_id2name = {0: '负面评价', 1: '正面评价'}

# 读取停用词（要和训练时保持一致）
with open("../data/processed/final_data/stopwords.txt", 'r', encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f.readlines()])


def segment_text(text):
    words = jieba.lcut(str(text))
    words = [w for w in words if w not in stop_words and w.strip()]
    return ' '.join(words)


# 2、定义预测函数
def predict(data):
    # 第一步：对输入数据进行分词
    words = segment_text(data["text"])

    # 第二步：用同一个向量化器把文本转成数值向量
    cat_feature = cat_tfidf.transform([words])
    sent_feature = sent_tfidf.transform([words])

    # 第三步：模型预测
    cat_pred = cat_model.predict(cat_feature)
    sent_pred = sent_model.predict(sent_feature)

    # 第四步：把数字标签翻译成类别名并返回
    data["predict_category"] = cat_id2name[cat_pred[0]]
    data["predict_sentiment"] = sent_id2name[sent_pred[0]]
    return data


if __name__ == '__main__':
    # 单独运行本文件时才做一次测试预测；被 api.py 导入时不会执行
    print(predict({"text": "手机信号很好，拍照清晰，非常满意！"}))
    print(predict({"text": "快递太慢了，包装也破了，失望。"}))

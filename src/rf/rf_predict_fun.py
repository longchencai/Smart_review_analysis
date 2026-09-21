# 导包
import jieba
from config import Config
import pickle

# 1.提前创建配置对象
config = Config()
# 2.加载tfidf(7分类一个,情感二分类一个)
with open(config.tfidf_save_path, 'rb') as f:
    tfidf = pickle.load(f)
with open(config.tfidf_sentiment_save_path, 'rb') as f:
    sentiment_tfidf = pickle.load(f)
# 3.加载rf模型(7分类一个,情感二分类一个)
with open(config.rf_save_model_path, 'rb') as f:
    model = pickle.load(f)
with open(config.rf_sentiment_save_model_path, 'rb') as f:
    sentiment_model = pickle.load(f)


# 4.定义api接口
def predict_fun(data):
    # todo 4.1 json数据中获取文本,然后直接做分词
    words = " ".join(jieba.lcut(data['text']))
    # todo 4.2 tfidf文本转数值特征
    number_words = tfidf.transform([words])
    # todo 4.3 rf模型预测类别索引
    y_pred_list = model.predict(number_words)  # [7]
    y_pred_idx = y_pred_list[0]
    # todo 4.4 根据索引获取类别名
    id2class = {index: line.strip() for index, line in enumerate(open(config.class_path, 'r', encoding='utf8'))}
    y_pred_class = id2class[y_pred_idx]

    # todo 4.5 情感二分类:同一份文本再用情感模型预测一次
    sentiment_number_words = sentiment_tfidf.transform([words])
    y_sentiment_idx = sentiment_model.predict(sentiment_number_words)[0]
    y_pred_sentiment = config.sentiment_id2class[y_sentiment_idx]

    # todo 4.6 拼接到data中并返回
    data['predict_class'] = y_pred_class
    data['predict_sentiment'] = y_pred_sentiment
    # 返回
    return data


if __name__ == '__main__':
    # 模拟页面传递过来json数据
    text = input('请您输入一条评论:')
    data = {"text": text}
    # 模拟调用API
    result = predict_fun(data)
    print(result)

# 导包
import jieba
from config import Config
import pickle

# 1.提前创建配置对象
config = Config()

# 2.模型与特征器改为【惰性加载】
#    原先这 4 个文件写在模块顶层直接 pickle.load，合计约 421 MB：
#      rf_model.pkl            ≈245 MB
#      rf_model_sentiment.pkl  ≈173 MB
#      两个 tfidf              ≈各 1.1 MB
#    后果是「只要 import 这个模块」就吃掉 421 MB 内存 —— 哪怕只想用其中一个任务，
#    或者只是被别的脚本间接导入。改成首次调用 predict_fun() 时才加载，导入即零成本。
#    4 个全局名保持存在（初值 None），外部若有引用仍能解析。
tfidf = None
sentiment_tfidf = None
model = None
sentiment_model = None
id2class = None
_loaded = False


def _load_models():
    """首次调用时加载模型与特征器，之后复用同一份。"""
    global tfidf, sentiment_tfidf, model, sentiment_model, id2class, _loaded
    if _loaded:
        return
    with open(config.tfidf_save_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(config.tfidf_sentiment_save_path, 'rb') as f:
        sentiment_tfidf = pickle.load(f)
    with open(config.rf_save_model_path, 'rb') as f:
        model = pickle.load(f)
    with open(config.rf_sentiment_save_model_path, 'rb') as f:
        sentiment_model = pickle.load(f)
    # 类别名映射只读一次（原实现是每次调用 predict_fun 都重新读一遍 class.txt）
    id2class = {index: line.strip()
                for index, line in enumerate(open(config.class_path, 'r', encoding='utf8'))}
    _loaded = True


# 4.定义api接口
def predict_fun(data):
    _load_models()
    # todo 4.1 json数据中获取文本,然后直接做分词
    words = " ".join(jieba.lcut(data['text']))
    # todo 4.2 tfidf文本转数值特征
    number_words = tfidf.transform([words])
    # todo 4.3 rf模型预测类别索引
    y_pred_list = model.predict(number_words)  # [7]
    y_pred_idx = y_pred_list[0]
    # todo 4.4 根据索引获取类别名（映射已在 _load_models 里缓存）
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

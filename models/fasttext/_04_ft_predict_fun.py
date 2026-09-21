# 导包
import fasttext
import jieba

from _01_config import Config

# TODO 提前创建配置对象
config = Config()

# TODO 加载模型（方案B：只需要加载这一个模型，就能同时预测商品大类和情感）
model = fasttext.load_model(config.ft_model_default)


# TODO 定义API接口函数
def predict_fun(data):
    # todo 1. json数据中获取文本,然后直接做分词
    words = " ".join(jieba.lcut(data['text']))
    # todo 2. fasttext模型直接预测（k=9 把全部标签都取回来，否则ova默认只返回概率>0.5的标签）
    y_pred_tuple = model.predict(words, k=9)
    print(y_pred_tuple)
    # todo 3. 按前缀拆分：c开头是商品大类，s开头是情感，各取概率最大的那个
    cat_name, sent_name = '', ''
    cat_p, sent_p = -1.0, -1.0
    for lb, p in zip(y_pred_tuple[0], y_pred_tuple[1]):
        raw = lb.replace('__label__', '')      # 去掉 __label__ 前缀
        if raw.startswith('c') and p > cat_p:
            cat_p, cat_name = float(p), config.id2class[int(raw[1:])]
        elif raw.startswith('s') and p > sent_p:
            sent_p, sent_name = float(p), config.id2sentiment[int(raw[1:])]
    # todo 4. 拼接到data中返回
    data['predict_class'] = cat_name                  # 商品大类
    data['predict_sentiment'] = sent_name             # 情感倾向
    data['class_score'] = round(cat_p, 4)             # 大类置信度
    data['sentiment_score'] = round(sent_p, 4)        # 情感置信度
    # 返回
    return data


if __name__ == '__main__':
    # 模拟准备json格式数据
    text = input('请您输入一个商品评论:')
    data = {"text": text}
    # TODO 模拟调用API
    result = predict_fun(data)
    print(result)

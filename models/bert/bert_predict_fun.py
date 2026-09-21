# 导包
from bert_config import Config
from bert_classifier_model import MyBertMultiTaskClassifier
import torch

# todo 1.提前创建配置对象
config = Config()
# todo 2.加载模型
bert_model = MyBertMultiTaskClassifier()
# 这里的 load_state_dict() 方法是加载模型参数, 而不是加载整个模型, 所以需要先创建模型对象, 再加载参数
bert_model.load_state_dict(torch.load(config.bert_classifier_model_save_path, map_location=config.device))
# TODO 添加到指定设备上
bert_model.to(config.device)
# 设置评估模式
bert_model.eval()


# todo 3.定义 api 接口
def predict_fun(data):
    # 获取文本
    text = data['text']
    # tokenizer 特征处理
    text_tensor = config.bert_tokenizer(
        text,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    # TODO 添加到指定设备上
    text_tensor = {k: v.to(config.device) for k, v in text_tensor.items()}
    # 模型预测拿到两个任务的 logits
    with torch.no_grad():
        cat_logits, sent_logits = bert_model(text_tensor)
    # argmax() 拿到最大分数对应的索引, 就是预测标签索引
    cat_pred_idx = torch.argmax(cat_logits, dim=-1).item()
    sent_pred_idx = torch.argmax(sent_logits, dim=-1).item()
    # 根据索引获取类别名
    cat_pred_class = config.id2cat[cat_pred_idx]
    sent_pred_class = config.id2sent[sent_pred_idx]
    # 拼接到 data 中返回
    data['predict_category'] = cat_pred_class
    data['predict_sentiment'] = sent_pred_class
    # 返回
    return data


if __name__ == '__main__':
    # 模拟准备 json 格式数据
    text = input('请您输入一条商品评论:')
    data = {"text": text}
    # TODO 模拟调用 API
    result = predict_fun(data)
    print(result)

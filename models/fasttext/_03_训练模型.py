# 导包
import fasttext
import jieba

from _01_config import Config

# TODO 提前创建配置对象
config = Config()
# TODO 1.模型训练（方案B：单模型联合，一个模型同时学商品大类 + 情感）
model = fasttext.train_supervised(
    input=config.ft_train,   # TODO 必须是相对路径，fasttext打不开含中文的绝对路径
    loss='ova',        # TODO 联合多标签必须用ova，默认的softmax会让两个任务抢概率
    dim=100,           # 词向量维度
    bucket=100000,     # TODO 默认2000000会训出800MB模型，这里调小到100000（约50MB）
    wordNgrams=2,      # 词级别数据用词bigram
    epoch=25,          # 训练轮数
    lr=0.5,            # 学习率
    verbose=2)
# TODO 2.模型保存
model.save_model(config.ft_model_default)
print('模型保存成功!')
print('===============================================================================')
# 3.模型加载 此处不需要加载,因为在同一个文件中,一起运行的,直接可以评估
# fasttext.load_model(config.ft_model_default)
# TODO 4.模型评估
# 注意：方案B是联合多标签，model.test() 返回的是"按样本平均"的精确率/召回率,
#   不是准确率，所以数值会比想象中小（实测约 0.97/0.49），这是正常的。
#   要商品大类和情感各自的准确率/精确率/召回率/F1，用 _05_模型评估.py 跑测试集。
result = model.test(config.ft_test)
# 打印结果                          (数据个数,精确率,召回率)
print(f"模型评估结果是:{result}")
print('===============================================================================')
# TODO 5.模型预测（k=9 把全部标签都取回来，再按c/s前缀拆成两个任务）
words = " ".join(jieba.lcut('键盘手感不错，屏幕很清晰，很满意'))
y_pred_tuple = model.predict(words, k=9)
print(f'模型预测结果是:{y_pred_tuple}')
# 从结果里拆出商品大类和情感
cat_name, sent_name = '', ''
cat_p, sent_p = -1.0, -1.0
for lb, p in zip(y_pred_tuple[0], y_pred_tuple[1]):
    raw = lb.replace('__label__', '')
    if raw.startswith('c') and p > cat_p:
        cat_p, cat_name = p, config.id2class[int(raw[1:])]
    elif raw.startswith('s') and p > sent_p:
        sent_p, sent_name = p, config.id2sentiment[int(raw[1:])]
print(f'商品大类 = {cat_name} ({cat_p:.4f})，情感倾向 = {sent_name} ({sent_p:.4f})')

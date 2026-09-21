# 导包
import os

import pandas as pd
import jieba
from config import Config

# 提前创建Config对象
config = Config()

# 类别名 -> 类别编号:class.txt 的行号就是 train.txt/val.txt/test.txt 里的编号
class_names = [line.strip() for line in open(config.class_path, 'r', encoding='utf8')]
class2id = {name: index for index, name in enumerate(class_names)}


# 定义处理数据的函数
def process_data(base_path, process_path):
    # 1.读取数据(本项目原始数据是 csv,带 BOM,所以要写 encoding='utf-8-sig')
    df_data = pd.read_csv(base_path, encoding='utf-8-sig')
    # 2.apply()对每一行进行分词并存储为一列
    df_data['words'] = df_data['review'].apply(lambda x: " ".join(jieba.lcut(x)))
    # 3.准备两个任务各自的标签列:cat_label 是7分类编号,sent_label 是情感二分类标签
    df_data[config.cat_label_col] = df_data['cat_l1'].map(class2id)
    df_data = df_data.rename(columns={'label': config.sentiment_label_col})
    # 4.校验一级大类都能在 class.txt 里找到,否则标签会变成 NaN,训练时报错很难查
    if df_data[config.cat_label_col].isnull().any():
        raise ValueError('有评论的一级大类不在 class.txt 中,请先重建 class.txt')
    # 5.保存数据
    os.makedirs(os.path.dirname(process_path), exist_ok=True)
    df_data[['words', config.cat_label_col, config.sentiment_label_col]].to_csv(
        process_path, sep='\t', index=False, header=True)


if __name__ == '__main__':
    # 处理训练数据
    process_data(config.train_path, config.process_train_path)
    # 处理验证数据
    process_data(config.dev_path, config.process_dev_path)
    # 处理测试数据
    process_data(config.test_path, config.process_test_path)

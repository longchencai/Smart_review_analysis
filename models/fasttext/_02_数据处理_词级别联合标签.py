# 导包
import pandas as pd
import jieba

from _01_config import Config

# todo 提前创建配置对象
config = Config()


# todo 定义处理数据的api函数（词级别 + 方案B联合标签）
def process_data(base_path, process_path):
    """
    把 csv 处理成 fasttext 要求的格式，方案B：一行写两个标签
        __label__c3 __label__s1 键盘 手感 不错
        c开头 = 商品大类(一级大类)，s开头 = 情感(0负面/1正面)
    """
    # 读取csv（列：review, label, cat_l1, cat_l2）
    df = pd.read_csv(base_path)
    with open(process_path, 'w', encoding='utf8') as fw:
        for text, cat, label in zip(df['review'], df['cat_l1'], df['label']):
            # 1.先去掉换行和制表符，避免一行的数据被拆成两行
            text = str(text).replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            # 2.分词
            words = " ".join(jieba.lcut(text))
            # 3.拼成两个标签（方案B的核心：两个任务的标签写在同一行）
            cat_id = config.class2id[cat]
            ft_line = '__label__c' + str(cat_id) + ' __label__s' + str(int(label)) + ' ' + words + '\n'
            # 4.写出到文件
            fw.write(ft_line)


if __name__ == '__main__':
    process_data(config.train_path, config.ft_train)
    process_data(config.dev_path, config.ft_dev)
    process_data(config.test_path, config.ft_test)
    print('数据处理完成!')

    # 打印前3行看看格式对不对
    print('=== train 前3行 ===')
    with open(config.ft_train, 'r', encoding='utf8') as fr:
        for i, line in enumerate(fr.readlines()):
            if i >= 3:
                break
            print(line.strip()[:100])

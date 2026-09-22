# 导包
import torch
import pandas as pd
from bert_config import Config

# todo 提前创建配置对象
config = Config()


# todo 加载原始数据
def load_data_list(csv_path):
    """
    :param csv_path: train.csv / val.csv / test.csv 的路径
    原始数据:  review,label,cat_l1,cat_l2
    :return: [(评论文本, 情感标签, 商品大类标签), ...]
    """
    # 提前创建一个列表, 用于存储 (评论, 情感标签, 商品大类标签) 元组
    data_list = []
    # 读取 csv
    # 显式指定 utf-8-sig：数据 CSV 由清洗脚本以 utf-8-sig 落盘（带 BOM），
    # 不指定的话部分 pandas 版本会把首列名读成 '\ufeffreview'，
    # 随后 row['review'] 抛 KeyError，且报错信息很难指向真正原因。
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 遍历每一行
    for _, row in df.iterrows():
        text = str(row['review'])
        # 情感标签：csv 中 label 列已经是 0/1
        sent_label = int(row['label'])
        # 商品大类标签：把字符串映射成整数
        cat_label = config.cat2id[str(row['cat_l1'])]
        # 封装成元组
        data_list.append((text, sent_label, cat_label))
    # 返回结果
    return data_list


# todo 自定义 dataset 类: 1 个继承 3 个重写
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


# todo 自定义 collate_fn 函数
def my_collate_fn(batch):
    """
    :param batch: 一批的 [(评论, 情感标签, 商品大类标签), ...]
    :return: tokenizer 处理后的特征张量, 情感标签张量, 商品大类标签张量
    """
    # todo 单独获取这一批数据中的所有 texts、sent_labels、cat_labels
    texts, sent_labels, cat_labels = zip(*batch)
    # todo 先把标签转换为张量
    batch_sent_labels_tensor = torch.tensor(sent_labels)
    batch_cat_labels_tensor = torch.tensor(cat_labels)
    # todo 再用 tokenizer 处理 texts, 最后转换为张量
    batch_texts_tensor = config.bert_tokenizer(
        texts,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    # todo 返回结果
    return batch_texts_tensor, batch_sent_labels_tensor, batch_cat_labels_tensor


def test_dataloader():
    data_list = load_data_list(config.test_path)  # 数据转换
    data_set = MyDataSet(data_list)  # __init__
    print(f"数据集条数:{len(data_set)}")  # 底层调用了 __len__
    # 封装 dataloader
    dataloader = torch.utils.data.DataLoader(data_set, config.batch_size, shuffle=True, collate_fn=my_collate_fn)
    print(f"按照每批{config.batch_size}条, 共分{len(dataloader)}批")
    # TODO collate_fn 调用时机: 是遍历 dataloader 的时候自动调用
    for batch_texts_tensor, batch_sent_labels_tensor, batch_cat_labels_tensor in dataloader:
        print(f'tokenizer 处理后特征 shape: {batch_texts_tensor["input_ids"].shape}, 维度：{batch_texts_tensor["input_ids"].dim()}')
        print(f"这一批情感标签: {batch_sent_labels_tensor}, shape: {batch_sent_labels_tensor.shape}")
        print(f"这一批商品大类标签: {batch_cat_labels_tensor}, shape: {batch_cat_labels_tensor.shape}")
        # 拿着 batch 前向传播
        break


# TODO 提前构建训练/验证/测试 3 个 dataloader
def build_all_dataloader():
    # 1.构建 train_dataloader
    data_list = load_data_list(config.train_path)
    data_set = MyDataSet(data_list)
    train_dataloader = torch.utils.data.DataLoader(data_set, config.batch_size, shuffle=True, collate_fn=my_collate_fn)
    # 2.构建 val_dataloader
    data_list = load_data_list(config.val_path)
    data_set = MyDataSet(data_list)
    val_dataloader = torch.utils.data.DataLoader(data_set, config.batch_size, shuffle=False, collate_fn=my_collate_fn)
    # 3.构建 test_dataloader
    data_list = load_data_list(config.test_path)
    data_set = MyDataSet(data_list)
    test_dataloader = torch.utils.data.DataLoader(data_set, config.batch_size, shuffle=False, collate_fn=my_collate_fn)
    # 4.返回
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    test_dataloader()

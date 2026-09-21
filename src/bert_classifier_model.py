# 导包
import torch
from bert_config import Config
from dataloader_utils import build_all_dataloader

# TODO 加载配置对象
config = Config()


# TODO 自定义多任务分类模型  1 个继承 2 个重写
class MyBertMultiTaskClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层使用 bert 的原始结构
        self.bert = config.bert_model
        # BERT 输出 768 维度，需要分别转成 7 分类（商品大类）和 2 分类（情感）
        self.cat_linear = torch.nn.Linear(config.bert_config.hidden_size, config.cat_class_num)
        self.sent_linear = torch.nn.Linear(config.bert_config.hidden_size, config.sent_class_num)

    def forward(self, x):
        # todo 预训练好的 bert 前向传播
        result = self.bert(**x)
        # todo 把 bert 的 768 维结果同时输入两个分类头
        # result['pooler_output'] 形状 (batch_size, 768)
        cat_logits = self.cat_linear(result['pooler_output'])   # (batch_size, 7)
        sent_logits = self.sent_linear(result['pooler_output']) # (batch_size, 2)
        # todo 返回两个任务的 logits
        return cat_logits, sent_logits


if __name__ == '__main__':
    # 准备数据
    train_dataloader, val_dataloader, test_dataloader = build_all_dataloader()
    # 准备模型
    my_bert_model = MyBertMultiTaskClassifier()
    print(my_bert_model)

    # 遍历 dataloader 的时候自动调用 collate_fn 处理数据
    for batch_texts_tensor, batch_sent_labels_tensor, batch_cat_labels_tensor in test_dataloader:
        # 前向传播
        cat_logits, sent_logits = my_bert_model(batch_texts_tensor)
        print(f"商品大类 logits shape: {cat_logits.shape}")
        print(f"情感 logits shape: {sent_logits.shape}")
        break

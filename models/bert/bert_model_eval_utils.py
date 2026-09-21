# 导包
import torch

from bert_config import Config
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  训练核心流程: 斌子法则 1212
#  1 个配置文件
config = Config()


def model_eval(val_dataloader, my_bert_model):
    """
    :param val_dataloader: 验证集的 dataloader
    :param my_bert_model: 训练好的模型
    :return: 商品大类任务的 (acc, pre, rec, f1), 情感任务的 (acc, pre, rec, f1)
    """
    #  2 个准备
    # 数据和模型传参
    # 准备模型 评估模式
    my_bert_model.eval()
    # TODO 确保模型也在指定设备上(和输入一致), 即使外部没搬过也能独立调用
    my_bert_model.to(config.device)
    with torch.no_grad():
        # 提前创建 4 个列表, 用于存储两个任务的所有预测和真实标签
        all_cat_pred_labels, all_cat_true_labels = [], []
        all_sent_pred_labels, all_sent_true_labels = [], []
        #  1 个遍历
        # 内层循环控制批次
        for index, (batch_texts_tensor, batch_sent_labels_tensor, batch_cat_labels_tensor) in enumerate(tqdm(val_dataloader), start=1):
            #  TODO 把数据放到指定设备上
            batch_sent_labels_tensor = batch_sent_labels_tensor.to(config.device)
            batch_cat_labels_tensor = batch_cat_labels_tensor.to(config.device)
            batch_texts_tensor = {k: v.to(config.device) for k, v in batch_texts_tensor.items()}
            #  2 个核心
            # 前向传播
            cat_logits, sent_logits = my_bert_model(batch_texts_tensor)
            # 累加每批的预测标签和真实标签
            all_cat_true_labels.extend(batch_cat_labels_tensor.tolist())
            all_sent_true_labels.extend(batch_sent_labels_tensor.tolist())
            # argmax 拿到预测标签
            cat_pred_labels = torch.argmax(cat_logits, dim=-1)
            sent_pred_labels = torch.argmax(sent_logits, dim=-1)
            all_cat_pred_labels.extend(cat_pred_labels.tolist())
            all_sent_pred_labels.extend(sent_pred_labels.tolist())

        # 计算商品大类任务的指标
        cat_acc = accuracy_score(all_cat_true_labels, all_cat_pred_labels)
        cat_pre = precision_score(all_cat_true_labels, all_cat_pred_labels, average='macro', zero_division=0)
        cat_rec = recall_score(all_cat_true_labels, all_cat_pred_labels, average='macro', zero_division=0)
        cat_f1 = f1_score(all_cat_true_labels, all_cat_pred_labels, average='macro', zero_division=0)

        # 计算情感任务的指标
        sent_acc = accuracy_score(all_sent_true_labels, all_sent_pred_labels)
        sent_pre = precision_score(all_sent_true_labels, all_sent_pred_labels, average='macro', zero_division=0)
        sent_rec = recall_score(all_sent_true_labels, all_sent_pred_labels, average='macro', zero_division=0)
        sent_f1 = f1_score(all_sent_true_labels, all_sent_pred_labels, average='macro', zero_division=0)

        #  返回方便打印日志
        return (cat_acc, cat_pre, cat_rec, cat_f1), (sent_acc, sent_pre, sent_rec, sent_f1)

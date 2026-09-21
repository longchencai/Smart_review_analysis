# 导包
import torch

from bert_config import Config
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  训练核心流程: 斌子法则 1212
#  1 个配置文件
config = Config()


def resolve_device(model, device=None):
    """
    决定评估时数据该放到哪个设备。

    优先用调用方显式传入的 device；否则跟随「模型参数当前所在设备」，
    而不是跟随 config.device。

    这一点对量化模型是硬性要求：动态量化后的模型只能跑在 CPU 上。
    如果还按 config.device（cuda）去搬数据，会直接报设备不一致错误。
    量化后 Embedding/Linear 被换成量化模块、不再是 Parameter，
    但 LayerNorm / pooler 仍是普通 Parameter，所以一般仍能取到设备；
    极端情况取不到就退回 CPU。
    """
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def model_eval(val_dataloader, my_bert_model, device=None, verbose=True):
    """
    :param val_dataloader: 验证集的 dataloader
    :param my_bert_model: 待评估的模型
    :param device: 显式指定评估设备；None 表示跟随模型参数所在设备
    :param verbose: 是否打印 tqdm 进度与指标日志
    :return: 商品大类任务的 (acc, pre, rec, f1), 情感任务的 (acc, pre, rec, f1)
    """
    #  2 个准备
    # 数据和模型传参
    # 准备模型 评估模式
    was_training = my_bert_model.training
    my_bert_model.eval()
    # TODO 设备跟随模型本身, 保证即使外部没搬过、或模型是量化模型也能独立调用
    target_device = resolve_device(my_bert_model, device)
    with torch.no_grad():
        # 提前创建 4 个列表, 用于存储两个任务的所有预测和真实标签
        all_cat_pred_labels, all_cat_true_labels = [], []
        all_sent_pred_labels, all_sent_true_labels = [], []
        #  1 个遍历
        # 内层循环控制批次
        if verbose:
            iterator = enumerate(tqdm(val_dataloader), start=1)
        else:
            iterator = enumerate(val_dataloader, start=1)
        for index, (batch_texts_tensor, batch_sent_labels_tensor, batch_cat_labels_tensor) in iterator:
            #  TODO 把数据放到目标设备上
            batch_sent_labels_tensor = batch_sent_labels_tensor.to(target_device)
            batch_cat_labels_tensor = batch_cat_labels_tensor.to(target_device)
            batch_texts_tensor = {k: v.to(target_device) for k, v in batch_texts_tensor.items()}
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

        if verbose:
            print(f"评估日志: 大类 准确率:{cat_acc:.4f} 精确率:{cat_pre:.4f} 召回率:{cat_rec:.4f} f1:{cat_f1:.4f}")
            print(f"评估日志: 情感 准确率:{sent_acc:.4f} 精确率:{sent_pre:.4f} 召回率:{sent_rec:.4f} f1:{sent_f1:.4f}")

        # 恢复调用前的模式, 避免外部调用者忘了切回 train
        if was_training:
            my_bert_model.train()

        #  返回方便打印日志
        return (cat_acc, cat_pre, cat_rec, cat_f1), (sent_acc, sent_pre, sent_rec, sent_f1)

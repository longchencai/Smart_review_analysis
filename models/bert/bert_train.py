# 导包
from bert_config import Config
from dataloader_utils import build_all_dataloader
from bert_classifier_model import MyBertMultiTaskClassifier
from bert_model_eval_utils import model_eval
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# TODO 训练核心流程: 斌子法则 14251
# todo 1 个配置文件
config = Config()
# TODO 打印当前运行的设备
print(f"当前运行的设备: {config.device}")


# 模型训练 API
def model_train():
    # todo 4 个准备
    # 准备数据
    train_dataloader, val_dataloader, test_dataloader = build_all_dataloader()
    # 准备模型
    my_bert_model = MyBertMultiTaskClassifier()
    my_bert_model.train()
    # TODO 把模型放到指定设备(GPU/CPU), 必须和 model_eval 里输入所在的设备一致, 否则会报设备不一致错误
    my_bert_model.to(config.device)
    # 准备损失函数：两个任务都用交叉熵
    cat_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    sent_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # 准备优化器
    optimizer = torch.optim.AdamW(my_bert_model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    # TODO 提前定义 best_score, 初始 0
    best_f1score = 0
    # todo 2 个遍历
    # 外层循环控制轮次
    for epoch in range(1, config.epochs + 1):
        # todo 额外添加日志变量  每 10 批打印损失, 准确率, 精确率, 召回率, f1 分数
        total_loss, batch_cnt = 0, 0
        all_cat_pred_labels, all_cat_true_labels = [], []
        all_sent_pred_labels, all_sent_true_labels = [], []
        # 内层循环控制批次
        for index, (batch_texts_tensor, batch_sent_labels_tensor, batch_cat_labels_tensor) in enumerate(tqdm(train_dataloader), start=1):
            # TODO 把数据也放到指定设备上, 必须和模型在同一个设备(否则 eval 时会设备不一致)
            batch_sent_labels_tensor = batch_sent_labels_tensor.to(config.device)
            batch_cat_labels_tensor = batch_cat_labels_tensor.to(config.device)
            batch_texts_tensor = {k: v.to(config.device) for k, v in batch_texts_tensor.items()}
            # todo 5 个核心
            # 前向传播
            cat_logits, sent_logits = my_bert_model(batch_texts_tensor)
            # 计算损失：两个任务损失相加
            cat_loss = cat_loss_fn(cat_logits, batch_cat_labels_tensor)
            sent_loss = sent_loss_fn(sent_logits, batch_sent_labels_tensor)
            loss = cat_loss + sent_loss
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # todo 额外计算日志变量
            total_loss += loss.item()
            batch_cnt += 1
            all_cat_true_labels.extend(batch_cat_labels_tensor.tolist())
            all_sent_true_labels.extend(batch_sent_labels_tensor.tolist())
            cat_pred_labels = torch.argmax(cat_logits, dim=-1)
            sent_pred_labels = torch.argmax(sent_logits, dim=-1)
            all_cat_pred_labels.extend(cat_pred_labels.tolist())
            all_sent_pred_labels.extend(sent_pred_labels.tolist())
            # todo 额外打印日志  每 10 批打印损失, 准确率, 精确率, 召回率, f1 分数
            if index % 10 == 0 or index == len(train_dataloader):
                # 计算商品大类任务指标
                cat_acc = accuracy_score(all_cat_true_labels, all_cat_pred_labels)
                cat_pre = precision_score(all_cat_true_labels, all_cat_pred_labels, average='macro')
                cat_rec = recall_score(all_cat_true_labels, all_cat_pred_labels, average='macro')
                cat_f1 = f1_score(all_cat_true_labels, all_cat_pred_labels, average='macro')
                # 计算情感任务指标
                sent_acc = accuracy_score(all_sent_true_labels, all_sent_pred_labels)
                sent_pre = precision_score(all_sent_true_labels, all_sent_pred_labels, average='macro')
                sent_rec = recall_score(all_sent_true_labels, all_sent_pred_labels, average='macro')
                sent_f1 = f1_score(all_sent_true_labels, all_sent_pred_labels, average='macro')
                # 计算损失
                avg_loss = total_loss / batch_cnt
                # todo 打印日志
                print(f"训练日志: 轮次:{epoch}, 当前批次:{index}, 损失:{avg_loss:.4f}")
                print(f"  商品大类 -> 准确率:{cat_acc:.4f}, 精确率:{cat_pre:.4f}, 召回率:{cat_rec:.4f}, f1:{cat_f1:.4f}")
                print(f"  情感 -> 准确率:{sent_acc:.4f}, 精确率:{sent_pre:.4f}, 召回率:{sent_rec:.4f}, f1:{sent_f1:.4f}")
                # 清空日志变量
                total_loss, batch_cnt = 0, 0
                all_cat_pred_labels, all_cat_true_labels = [], []
                all_sent_pred_labels, all_sent_true_labels = [], []
            # TODO 边训练边评估  每 100 批, 评估一次, 如果当前分数高于历史最高分, 记录并保存
            if index % 100 == 0 or index == len(train_dataloader):
                # 调用 model_eval() 进行评估
                (cat_acc, cat_pre, cat_rec, cat_f1), (sent_acc, sent_pre, sent_rec, sent_f1) = model_eval(val_dataloader, my_bert_model)
                # 综合 f1：两个任务 f1 的平均值
                avg_f1 = (cat_f1 + sent_f1) / 2
                print(f"评估日志: 轮次:{epoch}, 当前批次:{index}")
                print(f"  商品大类 -> 准确率:{cat_acc:.4f}, 精确率:{cat_pre:.4f}, 召回率:{cat_rec:.4f}, f1:{cat_f1:.4f}")
                print(f"  情感 -> 准确率:{sent_acc:.4f}, 精确率:{sent_pre:.4f}, 召回率:{sent_rec:.4f}, f1:{sent_f1:.4f}")
                print(f"  综合 f1:{avg_f1:.4f}")
                # 一定记得把模型再切换到训练模式
                my_bert_model.train()
                # 判断当前评估分数是否是最优分数, 如果是就记录并保存
                if avg_f1 > best_f1score:
                    # 记录当前最优分数
                    best_f1score = avg_f1
                    # todo 1 个保存
                    torch.save(my_bert_model.state_dict(), f"{config.bert_classifier_model_save_path}")
                    print(f"保存当前最优综合 f1:{avg_f1:.4f} 的模型")


if __name__ == '__main__':
    model_train()

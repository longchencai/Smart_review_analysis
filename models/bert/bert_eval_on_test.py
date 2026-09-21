# 导包
import torch
from bert_config import Config
from dataloader_utils import build_all_dataloader
from bert_classifier_model import MyBertMultiTaskClassifier
from bert_model_eval_utils import model_eval

# TODO 最终评估核心流程
# 加载训练时保存的「最优模型」，在 test 集上做最终、独立的指标报告
# 说明：train 用于更新参数；val 仅用于「边训练边评估、选最优模型」；
#       test 在整个训练过程中完全没碰过，训完之后才用它报最终成绩（避免数据泄漏）

# 1 个配置文件
config = Config()

# 准备数据：只取 test_dataloader（train/val 用不到）
_, _, test_dataloader = build_all_dataloader()

# 准备模型：重建结构 -> 加载最优权重
my_bert_model = MyBertMultiTaskClassifier()
my_bert_model.load_state_dict(
    torch.load(config.bert_classifier_model_save_path, map_location=config.device)
)
my_bert_model.to(config.device)

# 在 test 集上评估两个任务
(cat_acc, cat_pre, cat_rec, cat_f1), (sent_acc, sent_pre, sent_rec, sent_f1) = model_eval(
    test_dataloader, my_bert_model
)

# 打印最终结果
print("\n==== 最优模型在 TEST 集上的最终指标 ====")
print(f"商品大类 -> 准确率:{cat_acc:.4f}, 精确率:{cat_pre:.4f}, 召回率:{cat_rec:.4f}, f1:{cat_f1:.4f}")
print(f"情感     -> 准确率:{sent_acc:.4f}, 精确率:{sent_pre:.4f}, 召回率:{sent_rec:.4f}, f1:{sent_f1:.4f}")
print(f"综合 f1(平均): {(cat_f1 + sent_f1) / 2:.4f}")


if __name__ == '__main__':
    pass

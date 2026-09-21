# BERT 多任务分类模型（商品大类 + 情感）

## 任务
对中文商品评论做**双任务**分类（一个 BERT 编码器 + 两个独立分类头，一次前向同时出两个结果）：
- **商品大类**：7 分类（cat_l1）
- **情感正负面**：2 分类（label：0=负面，1=正面）

> 输入一条评论 → 同时输出 `predict_category`（大类）+ `predict_sentiment`（正/负）。

## 模型结构
- 预训练编码器：`bert-base-chinese`（共享，不随本项目发布，运行时从本地/缓存加载）
- 分类头1：`Linear(768, 7)` —— 商品大类
- 分类头2：`Linear(768, 2)` —— 情感
- 训练损失：`loss = cat_loss + sent_loss`，按综合 f1 保存最优

## 数据
- 来源：`data/processed/final_data/`（train / val / test）
- 划分比例：**70% / 15% / 15%**（train=43834，val=9393，test=9394）
- 标签：`review`（文本）、`cat_l1`（大类）、`label`（情感 0/1）

## 超参数
| 参数 | 值 |
|---|---|
| epochs | 5 |
| batch_size | 16 |
| max_len | 256 |
| lr | 5e-5 |
| optimizer | AdamW |

> max_len=256 依据：真实 BERT token 长度 P95≈182、P99≈359；256 覆盖 98% 样本、仅截断 2%，由数据分布 + jev 决策双重确认（见 `src/eda_maxlen_analysis.py`、`src/systemone_decide_maxlen.py`）。

## 评估指标
最终指标（test 集，未参与训练）见同目录 `eval_result.txt`。

## 文件说明
- `model/bert_multitask_classifier_model.pt`：模型权重（state_dict，由 Git LFS 管理）
- `eval_result.txt`：test 集 acc / pre / rec / f1（两个任务）
- 训练 / 评估 / 预测脚本：`src/bert_train.py`、`src/bert_eval_on_test.py`、`src/bert_predict_fun.py`
- 模型定义：`src/bert_classifier_model.py`；数据加载：`src/dataloader_utils.py`；工具：`src/bert_model_eval_utils.py`；配置：`src/config.py`

## 使用
```bash
conda activate nlp
cd src
python bert_predict_fun.py        # 单条评论推理
python bert_eval_on_test.py       # 在 test 集上复算指标
```

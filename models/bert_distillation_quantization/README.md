# BERT 蒸馏 + 动态量化（SmallBERT 4L/384 学生）

> 把已训练好的**双任务 BERT 教师**（商品大类 7 分类 + 情感二分类，102.27M 参数 / 390 MB）
> 蒸馏成小 BERT 学生，再做 int8 动态量化，全部在同一份验证集、同一套评估代码下对比。

---

## 一、结论摘要

| 模型 | 参数量 | 体积 | 设备 | 大类 macro-F1 | 情感 macro-F1 | 综合 F1 |
|---|---|---|---|---|---|---|
| 教师 BERT 12L/768 | 102,274,569 | 390.22 MB | GPU | 0.9208 | 0.9607 | 0.9407 |
| 基线学生（4 轮恒定 lr）| 15,560,457 | 59.39 MB | GPU | 0.8534 | 0.9068 | 0.8801 |
| **实验B（最优蒸馏）** | 15,560,457 | 59.39 MB | GPU | **0.8943** | 0.9202 | **0.9072** |
| **实验B + 动态量化** | 15,560,457 | **15.11 MB** | **CPU** | 0.8943 | 0.9204 | **0.9074** |

- **蒸馏提升 +2.71 个点**（0.8801 → 0.9072），对教师保留率 **96.44%**
- **量化几乎无损**：综合 F1 `+0.0002`
- **端到端压缩 390.22 MB → 15.11 MB = 25.8×**（蒸馏 6.57× × 量化 3.93×）
- 量化后 CPU 推理 25.94 → **18.55 ms/条（1.40× 加速）**

> **只想要「输入一条评价 → 输出类别 + 情感」？** 直接跳到第十一节「预测入口」，
> 用 `predict.py` 的 `StudentPredictor`，三行代码即可接入后端。

### 三组实验的增量（每组只比上一组多一个变量）

| 实验 | 改动 | 综合 F1 | Δ | 结论 |
|---|---|---|---|---|
| 基线 | 4 轮、恒定 lr=3e-4 | 0.8801 | — | 严重欠拟合 |
| **A** | + warmup/余弦调度 + 8 轮 | **0.9067** | **+0.0266** | ✅ **最大收益来自这里** |
| **B** | A + 逐层 hidden-state 蒸馏 | 0.9072 | +0.0005 | ⚠️ 几乎无增量 |
| **C** | B + 家用电器类别加权 | 0.8990 | **−0.0082** | ❌ **反而变差** |

---

## 二、目录与文件说明

```
models/bert_distillation_quantization/
├── bert_config.py               【改】唯一配置源；含单例 Config（见第九节坑#10）
├── student_model.py             【新】SmallBERT 4L/384 双任务学生定义
├── distill_data.py              【新】带教师软标签/[CLS]隐状态的 Dataset + 两个 collate_fn
├── cache_teacher_logits.py      【新】步骤 1：离线预计算教师监督信号
├── distill_train.py             【新】步骤 2：蒸馏训练，--exp A|B|C 切换三组实验
├── quantize.py                  【新】步骤 3：动态量化 + 量化后评估 + 加载自检
├── summarize_results.py         【新】解析 eval_result.txt 生成横向对比表
├── predict.py                   【新】★ 预测入口 StudentPredictor，供后端直接接入（见第十一节）
├── test_predict.py              【新】预测器测试（31 项，含端到端一致性校验）
├── report_utils.py              【新】评估与结果落盘，统一报告格式
├── bert_model_eval_utils.py     【改】修复 device 处理（量化评估必需）+ 静默模式
├── dataloader_utils.py          【改】支持自定义 batch_size、显式 utf-8-sig
├── bert_classifier_model.py     【原】教师模型定义（未改动）
├── bert_eval_on_test.py         【原】教师 test 集评估（未改动）
├── eval_result.txt              【产物】教师 + A/B/C + 量化 + 对比汇总
├── model/
│   ├── bert_multitask_classifier_model.pt   教师权重（390 MB，不入库）—— 蒸馏的知识来源与评估基准
│   ├── student_bert_4l384.pt                ★ 最优蒸馏学生 = 实验B（59 MB，入库）
│   └── student_bert_4l384_int8.pt           ★ 上者的动态量化版（15 MB，入库）
├── cache/
│   ├── teacher_soft_labels.npz              教师软标签 0.67 MB（不入库）
│   ├── teacher_cls_hidden.npz               教师[CLS]隐状态 237 MB（不入库）
│   └── eval_result_baseline_backup.txt      修正前基线结果备份（不入库）
└── bert-base-chinese/                       预训练骨架，权重不入库、配置与词表入库
```

> **只保留最优模型**：实验 A / C 的产物（各 59 MB）已在确认实验结论后清理。
> 需要时重跑 `distill_train.py --exp A` / `--exp C` 即可重新得到（各约 27 分钟）。
> 实验 B 的产物与 `student_bert_4l384.pt` **逐字节相同**（已用 MD5 校验：`4CCC7926...`），故不再单独保留。

---

## 三、环境要求（**两个环境分工不同，不能混用**）

| 环境 | conda 路径 | torch | CUDA | 用途 |
|---|---|---|---|---|
| GPU | `C:\Users\29011\.conda\envs\dl-gpu` | 2.13.0+cu126 | ✅ | 缓存预计算、蒸馏训练、fp32 评估 |
| CPU | `C:\Users\29011\.conda\envs\dl` | 2.13.0+cpu | ❌ | **动态量化**、int8 评估 |

**为什么量化必须用 CPU 环境**：PyTorch 的 int8 量化算子（`quantized::linear_dynamic` 等）**只有 CPU 实现，没有 CUDA kernel**。
把 CUDA 模型直接丢给 `quantize_dynamic` **不会报错**，但会产出一个 CPU / CUDA 都跑不了的废模型（静默陷阱）。

公共依赖：`transformers 5.16.1`、`scikit-learn`、`pandas`、`numpy`、`tqdm`。
（该版本 `transformers` 已移除 `BertConfig.position_embedding_type`，代码里已规避。）

**注意：这台机器的 shell 是 Windows PowerShell 5.1**（不是 pwsh 7），`Set-Content -Encoding utf8` 会写 BOM。
写需要给 git 读取的 UTF-8 文件时请用
`[System.IO.File]::WriteAllText($p, $s, (New-Object System.Text.UTF8Encoding($false)))`。

---

## 四、运行流程

```powershell
$dir = "C:\Users\29011\PycharmProjects\ai8_-project1\models\bert_distillation_quantization"
$gpu = "C:\Users\29011\.conda\envs\dl-gpu\python.exe"
$cpu = "C:\Users\29011\.conda\envs\dl\python.exe"

# 步骤 1：离线预计算教师监督信号（约 8 分钟；两份缓存都在时会跳过，重跑加 --force）
& $gpu -u "$dir\cache_teacher_logits.py"

# 步骤 2：蒸馏训练（三组实验，各自约 27 分钟）
& $gpu -u "$dir\distill_train.py" --exp A --reset --teacher-ref   # A 负责清空结果文件 + 评估教师
& $gpu -u "$dir\distill_train.py" --exp B
& $gpu -u "$dir\distill_train.py" --exp C

# 步骤 3：把最优实验提升为正式交付模型并量化（约 8 分钟）
Copy-Item "$dir\model\student_bert_4l384_expB.pt" "$dir\model\student_bert_4l384.pt" -Force
& $cpu -u "$dir\quantize.py" --model "$dir\model\student_bert_4l384.pt" `
                              --out "$dir\model\student_bert_4l384_int8.pt" --label "expB"

# 步骤 4：生成横向对比表并追加进 eval_result.txt
& $cpu -u "$dir\summarize_results.py" --append
```

所有超参都在 `bert_config.py` 里改，脚本不含硬编码超参；三组实验的差异由 `--exp` 控制在 `distill_train.EXPERIMENTS` 里声明。

---

## 五、关键设计决策

### 5.1 为什么离线缓存教师监督信号

教师冻结，对 43820 条只需前向一次。缓存后：

| | 在线蒸馏 | 离线缓存 |
|---|---|---|
| 显存 | 教师 + 学生 + 激活 | **只装学生** |
| 每 epoch | 教师前向 8.3 min + 学生训练 3.3 min | **只有学生 3.3 min** |
| 扫 T / α | 每次重跑教师 | **改超参秒级重跑** |

两份缓存都按 **CSV 原始行号**存取，所以 dataloader 可以放心 `shuffle=True`，对齐由下标保证。

- 软标签：只存**原始 logits**（不预先除以 T），扫温度时不必重跑教师
- `[CLS]` 隐状态：只对齐 `[CLS]` 位置。整条序列对齐需要
  `43820 × 256 × 768 × 4 层 ≈ 69 GB`，而 `[CLS]` 只需 **237 MB**。
  对分类任务而言 `[CLS]` 正是最终被分类头读的那个位置。

### 5.2 蒸馏损失：软标签和硬标签**都要**

```
loss = α·T²·KL(p_教师^T ‖ p_学生^T) + (1−α)·CE(y, p_学生) + hidden_weight·MSE
        └────────── 软标签项 ──────────┘   └──── 硬标签项 ────┘   └─ 实验B 才有 ─┘
```

| | 软标签（教师输出）| 硬标签（真实标注）|
|---|---|---|
| 提供什么 | **暗知识**：教师对各错误类的相对概率 | **地面真值** |
| 为什么不能省 | 没有它就不叫蒸馏 | 教师大类 F1 只有 0.9208，**约 8% 是错的**；纯软标签会让学生继承教师的错误 |
| 权重 | `α` | `1 − α` |

- **`T²` 只乘软标签项**：softmax 除以 T 会让梯度缩小约 `1/T²`，必须补偿；硬标签没经过温度缩放，**不能乘**。
- 双任务各算一份软损失、一份硬损失，再**等权平均**（与教师训练时 `cat_loss + sent_loss` 口径一致）。

### 5.3 ⚠️ `F.kl_div` 的参数顺序（易错点，别"修"错）

```python
F.kl_div(F.log_softmax(s_cat / T, -1),     # input  = 学生
         F.log_softmax(cat_soft / T, -1),  # target = 教师
         reduction='batchmean', log_target=True)
```

PyTorch 的 `F.kl_div(input, target)` 计算 `target·(log target − input)`，代入后即 **KL(教师‖学生)** ——
正是 Hinton 的交叉熵形式 `−Σ p_教师·log q_学生`（差一个常数 `H(教师)`）。
**看到"input 是学生"会本能觉得写反了，但这是对的。**

### 5.4 实验B：逐层 hidden-state 蒸馏

- **层映射**：学生第 1..4 层 ← 教师第 **3/6/9/12** 层（12 层按 3 倍等距抽样）
- **投影**：每层配一个可学习 `Linear(384→768)`，MSE 对齐 `[CLS]`
- **投影层只在训练期存在**，不进学生 `state_dict`，推理时丢弃（共 1,182,720 参数）
- 实测层索引：教师 `hidden_states` 有 13 项（索引 0 = embedding 输出，1..12 = 各层输出），学生有 5 项

### 5.5 实验C：类别加权（**实测为负收益**）

权重 = 逆频率，均值归一，上限 10：

| 类别 | 训练样本 | 权重 |
|---|---|---|
| 数码电子 | 11369 | 0.170 |
| 食品饮料 | 8413 | 0.229 |
| 个护美妆 | 6998 | 0.275 |
| 服饰鞋包 | 6996 | 0.275 |
| 本地生活 | 6968 | 0.277 |
| 图书文娱 | 2695 | 0.715 |
| **家用电器** | **381** | **5.059** |

只作用于硬标签交叉熵（软标签是分布匹配，不加权）。**结果见第七节：反而让家用电器 F1 掉了 10.5 个点。**

### 5.6 量化配置

```python
qconfig_spec = {
    torch.nn.Linear:    default_dynamic_qconfig,
    torch.nn.Embedding: float_qparams_weight_only_qconfig,   # ← 不能省
}
```

- 学生词嵌入占其参数量 **52.1%**（8,113,152 / 15,560,457）。**只量化 Linear 会丢掉一半以上压缩收益**（59.39 MB 只降到 38.7 MB）。
- `nn.Embedding` 用默认 qconfig 会直接 `AssertionError`，必须单独配 `float_qparams_weight_only_qconfig`。
- 本机 torch 构建**只提供 `onednn` 一个量化引擎**（`fbgemm` / `x86` 不可用）。

### 5.7 静态量化和 QAT 在本项目不可用

实测结论：**在 HuggingFace BERT 上根本跑不通**。

```
静态量化 PTQ (FX Graph Mode) : ❌ TypeError: slice indices must be integers or None or have an __index__ method
QAT (FX)                     : ❌ 同样错误
```

原因：FX symbolic tracing 用 `Proxy` 模拟张量，而 HF `BertModel` 里有大量依赖运行时数值的切片（如 `position_ids[:, :seq_length]`），
Python 的 slice 协议遇到 Proxy 直接抛错。这是 tracing 机制与 transformers 实现方式的根本冲突，不是配置问题。
因此**动态量化是唯一可行路径**。

---

## 六、超参速查（`bert_config.py`）

| 配置项 | 当前值 | 说明 |
|---|---|---|
| `student_layers` / `student_hidden` | `4` / `384` | 学生骨架 |
| `student_heads` / `student_ffn` | `6` / `1536` | 注意力头维度与教师同为 64 |
| `distill_T` / `distill_alpha` | `2.0` / `0.7` | 温度 / 软标签权重 |
| `distill_epochs` | `8` | 基线只跑 4 轮，欠拟合 |
| `distill_batch_size` | `32` | |
| `distill_lr` | `3e-4` | **随机初始化从零训练必须用预训练级 lr**，不能用 5e-5 |
| `distill_warmup_ratio` / `distill_schedule` | `0.1` / `cosine` | OneCycleLR：前 10% 升到 max_lr，之后余弦退火 |
| `distill_max_len` | `256` | 必须与教师一致，否则软标签分布不可比 |
| `distill_use_hidden_loss` | `False`（由 `--exp` 设置）| 实验B/C 置 True |
| `distill_hidden_weight` | `1.0` | 初始 MSE ≈ 0.99，与 CE/KL 量级可比，无需额外缩放 |
| `teacher_hidden_layer_map` | `(3, 6, 9, 12)` | 学生 1..4 层 ↔ 教师 3/6/9/12 层 |
| `distill_use_class_weight` | `False`（由 `--exp` 设置）| 实验C 置 True |
| `distill_class_weight_power` / `_max` | `1.0` / `10.0` | 逆频率 / 上限截断 |
| `eval_batch_size` | `64` | 仅影响速度，不影响指标 |
| `quantize_only_linear` | `False` | 设 `True` 会丢掉一半以上压缩收益 |
| `quant_engine` | `onednn` | 本机唯一可用引擎 |
| `csv_encoding` | `utf-8-sig` | 数据 CSV 带 BOM，必须显式指定 |

---

## 七、实验设计与结果（验证集 **9390** 条）

完整内容见 [`eval_result.txt`](eval_result.txt)。

### 7.1 主表

| 模型 | 体积 | 设备 | 大类 acc | 大类 macro-F1 | 情感 acc | 情感 macro-F1 | 综合 F1 |
|---|---|---|---|---|---|---|---|
| 教师 12L/768 | 390.22 MB | GPU | 0.9378 | 0.9208 | 0.9607 | 0.9607 | 0.9407 |
| 基线（4轮恒定lr）| 59.39 MB | GPU | 0.8915 | 0.8534 | 0.9068 | 0.9068 | 0.8801 |
| 实验A | 59.39 MB | GPU | 0.9133 | 0.8904 | 0.9230 | 0.9230 | 0.9067 |
| **实验B** | 59.39 MB | GPU | 0.9130 | **0.8943** | 0.9202 | 0.9202 | **0.9072** |
| 实验C | 59.39 MB | GPU | 0.9100 | 0.8771 | 0.9210 | 0.9210 | 0.8990 |
| **实验B int8** | **15.11 MB** | **CPU** | 0.9130 | 0.8943 | 0.9203 | 0.9204 | **0.9074** |

### 7.2 逐轮综合 F1（看出三组实验的收敛差异）

| 轮次 | A | B | C |
|---|---|---|---|
| 1 | 0.8266 | 0.8192 | **0.8474** |
| 2 | 0.8775 | 0.8682 | 0.8723 |
| 3 | 0.8908 | 0.8841 | 0.8841 |
| 4 | 0.8944 | 0.8903 | 0.8883 |
| 5 | 0.8975 | 0.8973 | 0.8920 |
| 6 | 0.9027 | 0.9058 | 0.8923 |
| 7 | 0.9064 | 0.9063 | **0.8990** |
| 8 | **0.9067** | **0.9072** | 0.8987 |

- **A 的收益几乎全部来自"调度 + 更多轮数"**：8 轮里每一轮都在涨，说明基线就是没训够。
- **B 前期落后 A、第 6 轮才反超**：hidden 目标在早期是竞争项，后期才转化为收益。
- **C 前期领先、中期被反超**：类别加权让模型很早就会"偏向稀有类"，但那并不等于学得更好。

### 7.3 逐类 F1（解释 C 为什么变差）

| 类别 | 样本 | 教师 | 基线 | A | B | C | C−B |
|---|---|---|---|---|---|---|---|
| 个护美妆 | 1499 | 0.8878 | 0.8302 | 0.8477 | 0.8440 | 0.8450 | +0.0010 |
| 图书文娱 | 578 | 0.9775 | 0.9299 | 0.9515 | 0.9552 | 0.9424 | −0.0129 |
| **家用电器** | **81** | 0.7867 | 0.5865 | 0.7218 | **0.7482** | **0.6433** | **−0.1049** |
| 数码电子 | 2436 | 0.9351 | 0.8862 | 0.9134 | 0.9103 | 0.9062 | −0.0042 |
| 服饰鞋包 | 1499 | 0.9134 | 0.8562 | 0.8884 | 0.8874 | 0.8913 | +0.0039 |
| 本地生活 | 1494 | 0.9895 | 0.9740 | 0.9777 | 0.9790 | 0.9804 | +0.0014 |
| 食品饮料 | 1803 | 0.9553 | 0.9111 | 0.9322 | 0.9356 | 0.9310 | −0.0047 |
| **macro** | 9390 | 0.9208 | 0.8534 | 0.8904 | **0.8943** | 0.8771 | **−0.0172** |

**核心发现：C 组本想救的「家用电器」，F1 反而掉了 10.5 个点。**
原因是加权只改变**决策边界的倾向**，不能补充信息 —— `weight=5.06` 让模型大量把样本判成家用电器，精确率崩塌。
**381 条训练样本的信息量上限，靠损失函数加权是突破不了的。** 其余 6 类变化都在 ±0.013 内，净效应由家用电器主导。

### 7.4 量化代价（实验B）

| 指标 | 基准 fp32 | 对照 int8 | 差值 |
|---|---|---|---|
| 大类 macro-F1 | 0.8943 | 0.8943 | `+0.0000` |
| 情感 macro-F1 | 0.9202 | 0.9204 | `+0.0002` |
| **综合 F1** | **0.9072** | **0.9074** | **`+0.0002`** |

| 其他 | 结果 |
|---|---|
| 体积 | 59.39 → 15.11 MB（25.5%，**3.93×**）|
| CPU 推理 | 25.94 → 18.55 ms/条（**1.40× 加速**）|
| 量化耗时 | **0.47 秒** |
| 加载自检 | `missing=[] unexpected=[]`，前向输出形状正确 ✅ |

---

## 八、已知问题与后续改进

### 8.1 还剩多少空间

教师 0.9407，学生 0.9072，**保留率 96.44%** —— 已进入 DistilBERT / TinyBERT 那一档（96~98%）。

按当前诊断，剩余空间主要来自：

1. **仍是欠拟合而非过拟合**。基线学生训练集 acc 0.9090 < 教师训练集 0.9353，train-val 差仅 1.75 点。
   加调度后指标一路涨到第 8 轮仍未饱和 → **再加轮数（12~16）可能还有收益**，这是最便宜的一档。
2. **宽度减半是硬约束**。教师 hidden=768，学生 384；文献里做得好的小 BERT 多数只减层数、不减宽度。
   若换成 **6L/512 或 4L/768**，精度应显著更高（代价是体积）。
3. **教师本身也未充分拟合**（训练 acc 0.9353 ≈ 验证 0.9378，几乎没有 gap），说明任务/数据的天花板大约在 0.94 附近。

### 8.2 已验证无效或负收益的手段

| 手段 | 结果 | 说明 |
|---|---|---|
| 类别加权 / focal loss | ❌ **−0.82 点** | 只改变决策边界倾向，无法补充 381 条样本的信息量 |
| 逐层 hidden-state 蒸馏 | ➖ **+0.05 点** | 在噪声范围内。可能因为只对齐了 `[CLS]`，而非 TinyBERT 的整条序列 |
| 教师权重初始化 | 未实施 | 经诊断后被降级（见 8.3）|

### 8.3 关于"用教师权重初始化学生"

README 早前版本曾把它列为第一优先级，**基于实测诊断已下调**：

- 诊断显示学生的瓶颈是**欠拟合**（训练集都拟合不动），不是"起点太差" → 增加训练量与监督信号的性价比高于搬运权重
- 三项搬运的可靠性分级：

| 措施 | 可靠性 | 风险 |
|---|---|---|
| 按头切片搬 QKV/O | 🟡 中 | 头是结构化的，但学生 hidden=384 残差流统计量不同，层配对敏感 |
| PCA 投影搬词嵌入 | 🟡 中 | ⚠️ BERT 词嵌入各向异性严重，前几个主成分常编码词频而非语义 |
| 按比例抽 4 层搬 FFN | 🔴 **低** | FFN 中间神经元**没有天然顺序**，任意切法很可能有害 |

### 8.4 下一步建议（按性价比排序）

| 优先级 | 措施 | 预期 |
|---|---|---|
| ⭐⭐⭐ | **epochs 提到 12~16**（沿用 OneCycle 或改成退火到 10% 而非 1%）| +0.5 ~ 1.5 点，成本仅算力 |
| ⭐⭐⭐ | 学生换 **6L/512** 或 **4L/768**（宽度别砍半）| +1 ~ 3 点，体积约 115 MB / 60 MB |
| ⭐⭐ | hidden 蒸馏改成**整条序列对齐**（需分块缓存或在线跑教师）| 可能让 B 从 +0.05 变成有意义的增量 |
| ⭐ | 给家用电器**补数据**（它只占 381 条，且教师自己也只有 F1 0.7867）| 这是唯一能真正救它的手段 |

### 8.5 消融实验缺口

当前只跑了 `T=2, α=0.7` 一组。README 早前建议的 α=0（纯硬标签）/ α=1（纯软标签）**尚未做** ——
有了这两条才能自证"软标签确实带来增益"，而不是仅凭引用文献。

---

## 九、踩坑记录

| # | 坑 | 现象 / 规避 |
|---|---|---|
| 1 | **量化加载顺序写反** | 把 fp32 权重灌进量化空壳 → `KeyError: '..._packed_params.dtype'`。<br>正确：<br>· 加载 **fp32** 权重 = 建 fp32 模型 → `load_state_dict` → 就地 `quantize_dynamic`<br>· 加载 **量化** 权重 = 先 `quantize_dynamic` 造同构空壳 → 再 `load_state_dict` |
| 2 | **在 CUDA 模型上量化** | **不报错**，但产出废模型：CPU 前向 `RuntimeError: apply_dynamic is not implemented for this packed parameter type`。必须先 `model.to('cpu')` |
| 3 | **对量化模型调 `.to('cuda')`** | 也**不报错**，错误延迟到前向：`NotImplementedError: Could not run 'quantized::linear_dynamic' ... 'CUDA' backend` |
| 4 | **只量化 `nn.Linear`** | 学生词嵌入占 52%，只量化 Linear 时体积几乎不降（59.39 → 38.7 MB）|
| 5 | **`nn.Embedding` 用默认 qconfig** | `AssertionError: Embedding quantization is only supported with float_qparams_weight_only_qconfig` |
| 6 | **蒸馏 lr 用了微调量级（5e-5）** | 4 个 epoch 后损失仍在快速下降、每轮指标都涨，严重欠拟合。随机初始化必须用 `3e-4` 量级 |
| 7 | **`transformers 5.16.1` 移除 `position_embedding_type`** | 构造 `BertConfig` 时 `AttributeError`，已从 `build_student_bert_config` 中删去 |
| 8 | **`pd.read_csv` 不指定编码** | 数据 CSV 带 UTF-8 BOM，理论上首列名会变成 `'\ufeffreview'`（实测 pandas 会自动跳过，但已显式写 `utf-8-sig`）|
| 9 | **`model_eval` 写死 `config.device`** | 量化模型只能跑 CPU，会报设备不一致。已改为 `next(model.parameters()).device` 跟随模型 |
| 10 | **每个模块各自 `config = Config()`** | `distill_train` 里 `apply_experiment('B')` 设的开关**对 `distill_data` 里那份 config 不可见** → dataloader 仍返回 5 元组、训练循环按 6 元组解包 → 实验B 首次运行崩溃。<br>**已把 `Config` 改成单例**（`__new__` + `_initialized` 守卫），全模块共享同一份配置，同时避免重复加载 BERT |
| 11 | **解析自己的报告时用 `split('【')`** | 段落副标题里也会出现 `【0】`、`【量化】` 这类引用，会把块从中间截断、静默丢掉整段。`summarize_results.py` 改成按**行首**的 `【` 切分（`re.split(r'\n(?=【)')`）|
| 12 | **PowerShell 5.1 的 `-Encoding utf8` 会写 BOM** | 导致 git 提交信息标题混入不可见 `U+FEFF`（`git log --format=%s` 首字符码点 0xFEFF）。改用 `[System.IO.File]::WriteAllText` + `UTF8Encoding($false)`，并用 `--amend` 修正 |
| 13 | **给后台脚本传参拼成单个字符串** | `& $py "$dir\x.py --force"` 被当成文件名 → `can't open file '...x.py --force'`。必须传数组 `@("$dir\x.py","--force")` |
| 14 | **量化模型上直接 `sum(model.parameters())` 数参数量** | 量化后 `nn.Linear`/`nn.Embedding` 不再是 `nn.Parameter`，实测只数到 **6,912**（真实 15,560,457）。`predict.py` 改为按 fp32 架构统计（`architectural_param_count()`）|
| 15 | **`predict_batch` 不分块** | 把整批一次性 tokenize + 前向，服务端收到上千条就会构造 (N, 256) 张量并 OOM。已改为按 `batch_size`（默认 64）内部自动分块 |

---

## 十、其他备忘

- **`torch.ao.quantization` 已被标记 deprecated**（官方建议迁移 `torchao`），torch 2.13 仍可用，脚本里已屏蔽重复警告。
- **量化不改变参数量**（仍是 15,560,457），改变的是存储精度与体积。
- **量化服务于无 GPU 的场景**：体积 −74.5%、CPU 提速 1.40×。若部署机器有 GPU，应直接用 fp32 学生（GPU 上更快）。
- 重新训练后 `distill_train.py --exp A --reset` 会**清空并重建** `eval_result.txt`，B/C 与 `quantize.py`、`summarize_results.py` 只**追加**，因此重跑不会累积历史垃圾段落。
- `cache/eval_result_baseline_backup.txt` 保留了修正前那一版的结果，`summarize_results.py` 会读它作为「基线」一行。

---

## 十一、预测入口（供后端接入）

`predict.py` 提供框架无关的 `StudentPredictor`：**输入一条评价文本 → 输出商品大类 + 情感**，
并附带置信度与候选类别。它不依赖任何 Web 框架，FastAPI / Flask / Django 都能直接调用。

### 11.1 三行接入

```python
from predict import get_predictor

predictor = get_predictor(quantized=True)          # 进程内单例，建议在服务启动时预热一次
result = predictor.predict("这个键盘手感很好，就是有点贵")
# {'text': '...', 'category': '数码电子', 'category_id': 3, 'category_confidence': 0.9982,
#  'sentiment': '正面评价', 'sentiment_id': 1, 'sentiment_confidence': 0.9481}
```

### 11.2 接口一览

| 方法 | 说明 |
|---|---|
| `StudentPredictor(quantized=True, device=None, model_path=None, top_k=0, batch_size=64)` | 构造。`quantized=True` 用 15 MB 的 int8 权重（强制 CPU）；`False` 用 59 MB 的 fp32 权重 |
| `.load()` | 显式加载，幂等。不调也行，首次预测会自动加载 |
| `.predict(text, top_k=None, with_truncation=False)` | 单条预测，返回 dict |
| `.predict_batch(texts, top_k=None, with_truncation=False, batch_size=None)` | 批量预测，**内部自动分块**，比逐条快得多 |
| `.info()` | 模型元信息（路径/精度/设备/参数量/类别表/加载耗时），适合做健康检查 |
| `get_predictor(...)` | 进程内单例，避免每个请求重复加载 |
| `predict(text, ...)` | 模块级便捷函数 |

返回字段：

```python
{
  "text": "...",                    # 原始输入
  "category": "数码电子",            # 商品大类（7 类之一）
  "category_id": 3,
  "category_confidence": 0.9982,    # softmax 概率
  "sentiment": "正面评价",           # 情感（正/负）
  "sentiment_id": 1,
  "sentiment_confidence": 0.9481,
  "top_categories": [...],          # 传 top_k>0 时返回，按概率降序
  "token_length": 23,               # 传 with_truncation=True 时返回
  "truncated": False,               # 同上；>256 token 会被截断
}
```

### 11.3 选 int8 还是 fp32

| | int8（默认）| fp32 |
|---|---|---|
| 模型体积 | **15.12 MB** | 59.39 MB |
| 设备 | **只能 CPU** | GPU / CPU |
| 吞吐（batch=64） | 19.2 ms/条 | **1.4 ms/条（GPU）** |
| 精度 | 综合 F1 0.9074 | 综合 F1 0.9072 |

**选型原则**：部署机器**没有 GPU** → int8（这也是量化的本意：换 CPU 成本，不是变快）；
**有 GPU** → fp32，GPU 上快 13 倍。

### 11.4 接后端时的注意事项

| 事项 | 说明 |
|---|---|
| **量化模型只能 CPU** | `quantized=True` 时 `device` 参数会被忽略并提示；不要试图把它搬到 GPU |
| **启动时预热** | 加载耗时约 1.0 秒（int8）/ 0.9 秒（fp32），在服务启动阶段调用 `get_predictor()` 即可 |
| **空文本会抛 `ValueError`** | 空串 / 纯空白 / 空列表都会被拒绝，请在 API 层映射成 422 或 400 |
| **批量要分块** | `predict_batch` 已内置分块（默认 64/批）；直接把上万条丢进去也不会 OOM |
| **int8 的置信度与 batch 组成有关** | 动态量化按当前 batch 的激活 min/max 现算 scale，实测置信度漂移约 2.5e-03，**预测标签不受影响**。若业务要求置信度可复现，固定 `batch_size=1` 或恒定批次 |
| **线程安全** | 模型 eval 模式 + `torch.no_grad()`，前向不改状态；HF tokenizer 是 Rust 实现也线程安全。多线程工作池可共享同一实例，无需加锁 |

### 11.5 命令行自测

```powershell
$py = "C:\Users\29011\.conda\envs\dl\python.exe"
& $py -u predict.py                                    # 内置 6 条样例（int8）
& $py -u predict.py --fp32 --text "手机拍照清晰但电池不耐用" --top-k 3
```

### 11.6 测试

`test_predict.py` 共 **31 项检查，全部通过**：

```powershell
& "C:\Users\29011\.conda\envs\dl-gpu\python.exe" -u test_predict.py
```

| 组 | 覆盖内容 |
|---|---|
| 1 加载与元信息 | int8 强制 CPU、fp32 跟随 config、参数量 15,560,457、体积、类别表、load 幂等 |
| 2 基本输出 | 字段齐全、类别/情感合法、置信度范围、JSON 可序列化、top_k 及降序 |
| 3 确定性与批量 | 同输入同输出、单条 vs 批量标签一致、分块漂移（int8 2.5e-03 / fp32 0）|
| 4 边界情况 | 空串/纯空白/空列表/批量含空 均正确报错；超长文本截断标记、短文本未截断 |
| 5 精度一致率 | fp32 vs int8 前 300 条：大类 **100%**、情感 **99.67%** |
| 6 **端到端一致性** | **预测器跑完整 9390 条验证集，fp32 复现 `0.8943/0.9202`、int8 复现 `0.8943/0.9204`，与 `eval_result.txt` 记录完全一致** → 证明预处理与离线评估严格对齐 |
| 7 单例 | 同参数返回同一实例、已预热 |

第 6 组是最有价值的一条：它把「离线指标」和「线上预测」钉死在同一套预处理上，
避免出现"评估 0.9072、上线却是另一回事"的经典事故。

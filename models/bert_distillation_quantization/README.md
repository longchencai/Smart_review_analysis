# BERT 蒸馏 + 动态量化（SmallBERT 4L/384 学生）

> 任务：把一个已训练好的**双任务 BERT 教师**（商品大类 7 分类 + 情感二分类，102.27M 参数 / 390 MB）
> 蒸馏成一个小 BERT 学生，再做 int8 动态量化，在**同一份验证集、同一套评估代码**下对比精度与体积。

---

## 一、结论摘要

| 模型 | 参数量 | 体积 | 设备 | 大类 macro-F1 | 情感 macro-F1 | 综合 F1 |
|---|---|---|---|---|---|---|
| 教师 BERT 12L/768 | 102,274,569 | 390.22 MB | GPU | 0.9208 | 0.9607 | **0.9407** |
| 纯蒸馏学生 4L/384 | 15,560,457 | 59.39 MB | GPU | 0.8534 | 0.9068 | **0.8801** |
| **蒸馏 + 动态量化 int8** | 15,560,457 | **15.12 MB** | **CPU** | **0.8537** | **0.9068** | **0.8802** |

- **端到端压缩：390.22 MB → 15.12 MB = 25.8×**（蒸馏 6.57× × 量化 3.93×）
- **量化几乎无损**：综合 F1 变化 `+0.0001`，逐项差值全在 ±0.0004 内
- **量化在 CPU 上更快**：25.10 → 18.48 ms/条（**1.36×**）
- 蒸馏**尚未达标**：学生对教师保留率仅 **93.56%**，原因见第八节「已知问题与后续改进」

---

## 二、目录与文件说明

```
models/bert_distillation_quantization/
├── bert_config.py               【改】唯一配置源：学生骨架 / 蒸馏超参 / 量化 / 路径 / 评估
├── student_model.py             【新】SmallBERT 4L/384 双任务学生定义
├── distill_data.py              【新】带教师软标签的 Dataset + collate_fn
├── cache_teacher_logits.py      【新】步骤 1：教师软标签离线预计算
├── distill_train.py             【新】步骤 2：双任务蒸馏训练（GPU）
├── quantize.py                  【新】步骤 3：动态量化 + 量化后评估 + 加载自检（CPU）
├── report_utils.py              【新】评估与结果落盘，统一报告格式
├── bert_model_eval_utils.py     【改】修复 device 处理（量化评估必需）+ 静默模式
├── dataloader_utils.py          【改】支持自定义 batch_size、显式 utf-8-sig
├── bert_classifier_model.py     【原】教师模型定义（未改动）
├── bert_eval_on_test.py         【原】教师 test 集评估（未改动）
├── eval_result.txt              【产物】四段评估结果 + 量化影响逐项差值
├── model/
│   ├── bert_multitask_classifier_model.pt   教师权重（390 MB，不入库）
│   ├── student_bert_4l384.pt                蒸馏学生 fp32（59 MB，入库）
│   └── student_bert_4l384_int8.pt           蒸馏+量化学生 int8（15 MB，入库）
├── cache/
│   └── teacher_soft_labels.npz              教师软标签（0.67 MB，不入库）
└── bert-base-chinese/                       预训练骨架，权重不入库、配置与词表入库
```

---

## 三、环境要求（**两个环境分工不同，不能混用**）

| 环境 | conda 路径 | torch | CUDA | 用途 |
|---|---|---|---|---|
| GPU | `C:\Users\29011\.conda\envs\dl-gpu` | 2.13.0+cu126 | ✅ | 软标签预计算、蒸馏训练、fp32 评估 |
| CPU | `C:\Users\29011\.conda\envs\dl` | 2.13.0+cpu | ❌ | **动态量化**、int8 评估 |

**为什么量化必须用 CPU 环境**：PyTorch 的 int8 量化算子（`quantized::linear_dynamic` 等）**只有 CPU 实现，没有 CUDA kernel**。
把 CUDA 模型直接丢给 `quantize_dynamic` **不会报错**，但会产出一个 CPU / CUDA 都跑不了的废模型（静默陷阱）。
量化结果也只能在 CPU 上推理。

公共依赖：`transformers 5.16.1`、`scikit-learn`、`pandas`、`numpy`、`tqdm`。
（注意：该版本 `transformers` 已移除 `BertConfig.position_embedding_type`，代码里已规避。）

---

## 四、三步运行

```powershell
$dir = "C:\Users\29011\PycharmProjects\ai8_-project1\models\bert_distillation_quantization"
$gpu = "C:\Users\29011\.conda\envs\dl-gpu\python.exe"
$cpu = "C:\Users\29011\.conda\envs\dl\python.exe"

# 步骤 1：教师软标签离线预计算（43820 条，约 8 分钟；已缓存时直接跳过，重跑加 --force）
& $gpu -u "$dir\cache_teacher_logits.py"

# 步骤 2：蒸馏训练（4 轮约 13 分钟 + 教师参考评估）
#         每次运行会清空并重建 eval_result.txt
& $gpu -u "$dir\distill_train.py"

# 步骤 3：动态量化 + 量化后评估（约 7 分钟）；只追加到 eval_result.txt
& $cpu -u "$dir\quantize.py"
```

所有超参都在 `bert_config.py` 里改，脚本不含硬编码超参。

---

## 五、关键设计决策

### 5.1 为什么离线缓存软标签

教师是冻结的，对训练集只需前向一次。缓存后：

| | 在线蒸馏 | 离线软标签 |
|---|---|---|
| 显存 | 教师 + 学生 + 激活 | **只装学生** |
| 每 epoch | 教师前向 + 学生前向 + 反向 | **只有学生** |
| 扫 T / α | 每次重跑教师 | **改超参秒级重跑** |

缓存只存**原始 logits**（`float16`），**不预先除以 T**——这样扫温度时不必重跑教师（8 分钟省下来）。
软标签按 **CSV 原始行号**存取，所以 dataloader 可以放心 `shuffle=True`，对齐由下标保证。

### 5.2 蒸馏损失：软标签和硬标签**都要**

```
loss = α · T² · KL(p_教师^T ‖ p_学生^T)  +  (1 − α) · CE(y, p_学生)
        └────────── 软标签项 ──────────┘      └────── 硬标签项 ──────┘
```

| | 软标签（教师输出） | 硬标签（真实标注） |
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

PyTorch 的 `F.kl_div(input, target)` 计算的是 `target·(log target − input)`，
代入后即 **KL(教师‖学生)** —— 正是 Hinton 的交叉熵形式 `−Σ p_教师·log q_学生`（差一个常数 `H(教师)`）。
**看到"input 是学生"会本能觉得写反了，但这是对的，调换方向会跑偏。**

### 5.4 量化配置

```python
qconfig_spec = {
    torch.nn.Linear:    default_dynamic_qconfig,
    torch.nn.Embedding: float_qparams_weight_only_qconfig,   # ← 不能省
}
```

- 学生词嵌入占其参数量 **52.1%**（8,113,152 / 15,560,457）。**只量化 Linear 会丢掉一半以上压缩收益**（59.39 MB 只降到 38.7 MB）。
- `nn.Embedding` 用默认 qconfig 会直接 `AssertionError`，必须单独配 `float_qparams_weight_only_qconfig`。
- 本机 torch 构建**只提供 `onednn` 一个量化引擎**（`fbgemm` / `x86` 不可用）。

### 5.5 静态量化和 QAT 在本项目不可用

实测结论：**在 HuggingFace BERT 上根本跑不通**。

```
静态量化 PTQ (FX Graph Mode) : ❌ TypeError: slice indices must be integers or None or have an __index__ method
QAT (FX)                     : ❌ 同样错误
```

原因：FX symbolic tracing 用 `Proxy` 模拟张量，而 HF `BertModel` 里有大量依赖运行时数值的切片（如 `position_ids[:, :seq_length]`），Python 的 slice 协议遇到 Proxy 直接抛错。
这是 tracing 机制与 transformers 实现方式的根本冲突，不是配置问题。因此**动态量化是唯一可行路径**。

---

## 六、超参速查（`bert_config.py`）

| 配置项 | 当前值 | 说明 |
|---|---|---|
| `student_layers` / `student_hidden` | `4` / `384` | 学生骨架 |
| `student_heads` / `student_ffn` | `6` / `1536` | 注意力头维度与教师同为 64 |
| `distill_T` | `2.0` | 温度 |
| `distill_alpha` | `0.7` | 软标签权重 |
| `distill_epochs` | `4` | |
| `distill_batch_size` | `32` | |
| `distill_lr` | `3e-4` | **随机初始化从零训练必须用预训练级 lr**，不能用 5e-5 |
| `distill_max_len` | `256` | 必须与教师一致，否则软标签分布不可比 |
| `distill_weight_decay` / `distill_grad_clip` | `0.01` / `1.0` | |
| `eval_batch_size` | `64` | 仅影响速度，不影响指标 |
| `quantize_only_linear` | `False` | 设 `True` 会丢掉一半以上压缩收益 |
| `quant_engine` | `onednn` | 本机唯一可用引擎 |
| `csv_encoding` | `utf-8-sig` | 数据 CSV 带 BOM，必须显式指定 |

---

## 七、实测结果（验证集 **9390** 条）

完整内容见 [`eval_result.txt`](eval_result.txt)，以下为摘要。

### 7.1 四段评估

| # | 模型 | 体积 | 设备 | 大类 acc / F1 | 情感 acc / F1 | 综合 F1 |
|---|---|---|---|---|---|---|
| 1 | 教师 BERT 12L/768 | 390.22 MB | GPU | 0.9378 / 0.9208 | 0.9607 / 0.9607 | 0.9407 |
| 2 | 纯蒸馏学生 fp32 | 59.39 MB | GPU | 0.8915 / 0.8534 | 0.9068 / 0.9068 | 0.8801 |
| 3 | 纯蒸馏学生 fp32 | 59.39 MB | CPU | 0.8915 / 0.8534 | 0.9068 / 0.9068 | 0.8801 |
| 4 | 蒸馏 + 量化 int8 | **15.12 MB** | CPU | 0.8918 / 0.8537 | 0.9068 / 0.9068 | **0.8802** |

> 【2】与【3】是同一次训练的两份权重，只是评估设备不同，指标完全一致 —— 可用来确认 fp32 数值在 GPU / CPU 上无差异。

### 7.2 量化影响（【4】−【3】）

| 指标 | 基准 fp32 | 对照 int8 | 差值 |
|---|---|---|---|
| 大类 准确率 | 0.8915 | 0.8918 | `+0.0003` |
| 大类 macro-F1 | 0.8534 | 0.8537 | `+0.0002` |
| 情感 准确率 | 0.9068 | 0.9068 | `+0.0000` |
| 情感 macro-F1 | 0.9068 | 0.9068 | `-0.0000` |
| **综合 F1** | **0.8801** | **0.8802** | **`+0.0001`** |

| 其他 | 结果 |
|---|---|
| 体积 | 59.39 → 15.12 MB（25.5%，**3.93×**）|
| CPU 推理 | 25.10 → 18.48 ms/条（**1.36× 加速**）|
| 量化耗时 | **0.42 秒** |
| 加载自检 | `missing=[] unexpected=[]`，前向输出形状正确 ✅ |

---

## 八、已知问题与后续改进

### 8.1 蒸馏尚未达标（保留率 93.56%）

**根本原因：学生是随机初始化、从零训练的。**

学生没有继承 BERT 的任何预训练语言知识，8.11M 个词嵌入参数（占其 52%）全靠 43.8k 样本从头学。
`student_model.py` 顶部已注明：宽度从 768 缩到 384 后教师权重矩阵形状不匹配，需要额外投影或按头切片才能搬运（TinyBERT 那一类做法），本实现暂未做。

### 8.2 大类 macro-F1 被稀有类拖垮

```
大类 accuracy = 0.8915   大类 macro-F1 = 0.8534   ← 差 3.8 个点
情感 accuracy = 0.9068   情感 macro-F1 = 0.9068   ← 无差距（二分类且均衡）
```

`家用电器`（热水器）训练集只有 381 条，其 F1 很低，把 macro 平均拉了下来。**这是数据分布问题，不是代码问题。**

### 8.3 教师本身也未充分拟合

用缓存软标签反推教师**训练集**准确率：大类 `0.9353`、情感 `0.9641` —— 与其验证集表现（`0.9378` / `0.9607`）几乎相同，**几乎没有过拟合 gap**。
说明该任务在当前数据上的天花板大致就在 0.94 附近，学生要追平并不容易。

### 8.4 改进清单（按性价比排序）

| 优先级 | 措施 | 预期 |
|---|---|---|
| ⭐⭐⭐ | **用教师权重初始化学生**：按头切片搬 QKV/O（取前 6 头）、PCA 投影搬词嵌入、按比例抽 4 层搬 FFN | 最大杠杆，可能 +3~5 个点 |
| ⭐⭐⭐ | 加 **warmup + 余弦衰减**，epochs 提到 8~10 | +1~2 个点，成本仅算力 |
| ⭐⭐ | 给 `家用电器` 加**类权重**或改用 focal loss | 直接拉升大类 macro-F1 |
| ⭐⭐ | 学生换成 **6L/512**（fp32 115 MB / int8 29 MB）| 精度更高，仍能进仓库 |
| ⭐ | 加 **hidden-state 蒸馏**（学生 4 层对齐教师第 3/6/9/12 层，需 384→768 投影）| TinyBERT 核心技巧 |

### 8.5 消融实验建议

当前只跑了 `T=2, α=0.7` 一组。建议补三条曲线（用 **val** 选，test 只在最后碰一次）：

| 组 | α | T | 用途 |
|---|---|---|---|
| A | 0 | — | 纯硬标签（普通监督训练）对照组 |
| B | 1 | 2 | 纯软标签，可直观展示"继承教师错误"现象 |
| C | 0.7 | 2 | 完整蒸馏（当前配置）|

有了 A 组才**能证明蒸馏有效**，而不是仅凭引用文献。

---

## 九、踩坑记录

| # | 坑 | 现象 / 规避 |
|---|---|---|
| 1 | **量化加载顺序写反** | 把 fp32 权重灌进量化空壳 → `KeyError: '..._packed_params.dtype'`。<br>正确：<br>· 加载 **fp32** 权重 = 建 fp32 模型 → `load_state_dict` → 就地 `quantize_dynamic`<br>· 加载 **量化** 权重 = 先 `quantize_dynamic` 造同构空壳 → 再 `load_state_dict` |
| 2 | **在 CUDA 模型上量化** | **不报错**，但产出废模型：CPU 前向 `RuntimeError: apply_dynamic is not implemented for this packed parameter type`，CUDA 前向 `NotImplementedError`。必须先 `model.to('cpu')` |
| 3 | **对量化模型调 `.to('cuda')`** | 也**不报错**，错误延迟到前向：`NotImplementedError: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend` |
| 4 | **只量化 `nn.Linear`** | 学生词嵌入占 52%，只量化 Linear 时体积几乎不降（59.39 → 38.7 MB）|
| 5 | **`nn.Embedding` 用默认 qconfig** | `AssertionError: Embedding quantization is only supported with float_qparams_weight_only_qconfig` |
| 6 | **蒸馏 lr 用了微调量级（5e-5）** | 4 个 epoch 后损失仍在快速下降、每轮指标都涨，严重欠拟合。随机初始化必须用 `3e-4` 量级 |
| 7 | **`transformers 5.16.1` 移除 `position_embedding_type`** | 构造 `BertConfig` 时 `AttributeError`，已从 `build_student_bert_config` 中删去 |
| 8 | **`pd.read_csv` 不指定编码** | 数据 CSV 带 UTF-8 BOM，理论上首列名会变成 `'\ufeffreview'`（实测 pandas 会自动跳过，但不要依赖，已显式写 `utf-8-sig`）|
| 9 | **`model_eval` 写死 `config.device`** | 量化模型只能跑 CPU，会报设备不一致。已改为 `next(model.parameters()).device` 跟随模型 |

---

## 十、其他备忘

- **`torch.ao.quantization` 已被标记 deprecated**（官方建议迁移到 `torchao`），torch 2.13 仍可用，脚本里已屏蔽重复警告。
- **量化不改变参数量**（仍是 15,560,457），改变的是存储精度与体积。
- **量化服务于无 GPU 的场景**：体积 −74.5%、CPU 提速 1.36×。若部署机器有 GPU，应该直接用 fp32 学生（GPU 上更快）。
- 重新训练后 `distill_train.py` 会**清空并重建** `eval_result.txt`，`quantize.py` 只**追加**，因此重跑不会累积历史垃圾段落。

# BERT 蒸馏 + 动态量化模块 —— 模型文件审查报告

- **审查对象**：`models/bert_distillation_quantization/`
- **审查日期**：2026-09-22
- **审查方式**：静态代码审查 + 权重文件实测 + 真实数据端到端复现 + 性能基准
- **实测环境**：`D:\conda_envs\PYTHON_ML`（Python 3.12.13 / torch 2.13.0+cpu / transformers 5.16.1 /
  scikit-learn 1.9.0），量化引擎 `onednn`（本机唯一可用），`torch.get_num_threads()=16`，逻辑核 32

---

## 一、结论速览

> **§ 状态更新（2026-09-22 上午，演示前）**
> 问题 1（基座权重缺失）**已修复**：已从 hf-mirror 下载 `model.safetensors`（392.49 MiB）
> 放入 `bert-base-chinese/`（该路径已被 `.gitignore:161` 的 `*.safetensors` 忽略，不污染仓库）。
> 修复后 **14 / 14 个模块全部导入成功**，`predict.py` 实测可跑通、6 条样例全部分类正确。
> 同时问题 7 的推断**曾被实测推翻，已在下文更正**（学生 test 集实测成绩并不低于 val）。

| 维度 | 结论 | 关键数字 |
|---|---|---|
| 权重数值 | **干净，无异常** | 无 NaN / Inf / 全零 / LayerNorm 退化 |
| 文件完整性 | 两个学生 `.pt` 完整；配套文件缺失（**基座已补齐，教师权重与缓存仍缺**） | 两个 `.pt` 均可 `strict=True` 加载 |
| 量化精度损失 | **极小，几乎无损** | Linear 相对误差 0.54%~1.54%，余弦相似度 >0.9998；词嵌入 0.666% |
| 压缩收益 | **已达 int8 理论上限** | 59.39 → 15.12 MiB，3.93x（理论 4x） |
| 推理性能 | **存在巨大浪费，可再快 4~9 倍** | 固定 padding 到 256 浪费 76.1% 计算 |
| 模型结构 | 基本合理，有一处明显冗余 | 词表 69.4% 的行从未被使用 |
| 蒸馏配置 | 参数合理，但缺少关键消融 | 唯一一组 T=2 / α=0.7，无 α=0/1 对照 |
| 可运行性 | **已修复**：14/14 模块可导入，推理入口实测跑通 | 修复前 13/14 报 `OSError` |
| 泛化（test 集） | **优于预期：与 val 基本持平，无明显落差** | 学生 test 综合 F1 **0.9082(fp32) / 0.9086(int8)** |

**一句话总结**：模型本身做得很扎实（量化几乎无损、压缩到理论上限、指标可 100% 复现、
**测试集成绩与验证集持平**），存在两类问题：① 交付件不完整（基座权重已补，教师权重与缓存仍缺，
导致蒸馏/教师评估链路无法复现）；② 输入处理方式浪费了四分之三以上的算力，
改一行 padding 策略就能快 4~9 倍。

---

## 二、问题清单

### P0 —— 阻断级：拿到这份代码无法运行

#### 问题 1：基座 BERT 权重缺失，导致 13/14 个模块在导入阶段就崩溃

> ✅ **已修复（2026-09-22 上午）**：已从 hf-mirror 下载 `model.safetensors`（392.49 MiB）到
> `bert-base-chinese/`。修复后 **14/14 模块导入成功**，`predict.py` 实测跑通。
> 下面保留原始诊断记录，供换机器 / 重新 clone 时参考。

**证据**

`bert-base-chinese/` 目录只有 6 个文件、合计 592 KB，**没有任何权重文件**：

```
README.md (2.1 KB)  config.json (649 B)  configuration.json (42 B)
tokenizer.json (460 KB)  tokenizer_config.json (32 B)  vocab.txt (130 KB)
→ 是否存在 pytorch_model.bin / model.safetensors：无
```

实测在全新进程中逐个导入模块：

| 模块 | 结果 |
|---|---|
| `bert_config` | OK（它只定义类，不实例化） |
| `bert_classifier_model` / `bert_model_eval_utils` / `dataloader_utils` / `report_utils` | **FAIL: OSError** |
| `student_model` / `distill_data` / `cache_teacher_logits` / `distill_train` / `quantize` | **FAIL: OSError** |
| `summarize_results` / `predict` / `test_predict` / `bert_eval_on_test` | **FAIL: OSError** |

报错原文：

```
OSError: Error no file named model.safetensors, or pytorch_model.bin,
         found in directory ...\bert-base-chinese.
```

四个环境变量完全一致的依赖（`PYTHON_ML`），14 个 `.py` 文件全部通过 `py_compile` 语法检查，
所以这**不是代码问题，而是文件缺失问题**。

**原因**

`bert_config.py:64-76` 的兜底候选链只检查 `config.json` 是否存在：

```python
for cand in model_candidates:
    if os.path.exists(cand + '/config.json'):   # ← 只检查配置，没检查权重
        model_path = cand
        break
```

这个目录里 `config.json` 存在但权重不存在，于是 `model_path` 被锁定到本地空目录，
永远走不到第 ③ 档「从 HuggingFace 下载」。而 `config.bert_model = BertModel.from_pretrained(...)`
又是在 `Config.__init__` 里无条件执行的，所以所有调用 `Config()` 的模块都倒在这里。

**影响**：`predict.py` 是 README 第十一节明确推荐的「后端三行接入」入口，当前**无法启动**。

**修复建议**

```python
WEIGHT_FILES = ('pytorch_model.bin', 'model.safetensors')

def _has_weights(d):
    return any(os.path.exists(os.path.join(d, w)) for w in WEIGHT_FILES)

model_path = None
for cand in model_candidates:
    if os.path.exists(cand + '/config.json') and _has_weights(cand):
        model_path = cand
        break
if model_path is None:
    model_path = 'bert-base-chinese'   # 交给 HF 自动下载（或指向本地 HF 缓存）
```

另外建议在 README 第三节「环境要求」里补一条前置命令，明确基座与教师权重的获取方式。
**本次实际生效的命令**（已验证，约 14 分钟，走 hf-mirror 镜像）：

```python
# 一键补齐基座权重（只补权重文件，不动已有的 config/tokenizer）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="bert-base-chinese", filename="model.safetensors",
                local_dir=r"models\bert_distillation_quantization\bert-base-chinese")
```

注意：`bert-base-chinese/*.safetensors` 已被 `.gitignore:161` 忽略，
所以下载进来的权重**不会污染仓库**，可以放心执行。脚本见
`docs/model_audit/fetch_base.py`。

---

#### 问题 2：单例初始化失败后被「半初始化」污染，把真因掩盖成看不懂的 AttributeError

**证据**

同一进程里连续两次调用 `Config()`：

```
第1次 Config():  -> 抛错 OSError
第2次 Config():  -> 竟然没抛错！
                -> _initialized      = True
                -> 有 bert_tokenizer 吗: False
                -> 有 cat2id 吗       : False
                -> 有 distill_max_len 吗: False
```

**原因**

`bert_config.py:34-37`：

```python
def __init__(self):
    if getattr(self, '_initialized', False):
        return
    self._initialized = True          # ← 在最开头就打上「已初始化」标记
    print('正在初始化配置文件....')
    ...                               # 后面任何一步失败，标记已经留下了
```

`__new__` 已经把 `_singleton` 赋值，`_initialized` 又是**先置位、后干活**，
所以第一次失败后，这个半成品对象会永久缓存下来。后续报错会变成：

```
AttributeError: 'Config' object has no attribute 'bert_tokenizer'
```

**影响**：排查「权重缺失」这件事会被引导到「tokenizer 为什么没有」的歧路上，
真因（`OSError`）只在第一次出现且缺少上下文。

**修复建议**（改 3 行）

```python
def __init__(self):
    if getattr(self, '_initialized', False):
        return
    print('正在初始化配置文件....')
    try:
        ...                            # 原有全部初始化逻辑
    except Exception:
        type(self)._singleton = None   # 失败就丢弃这个半成品
        raise
    self._initialized = True           # 全部成功后才置位
```

---

### P0 —— 阻断级：交付件不完整（与文档记载不符）

#### 问题 3：教师权重与教师缓存文件缺失，蒸馏链路与 test 评估无法复现

**证据**

`model/` 目录实际只有 2 个文件：

| 文件 | 大小 | 状态 |
|---|---|---|
| `student_bert_4l384.pt` | 62,273,746 B (59.39 MiB) | 存在，MD5 `4CCC792676306C727DEB41BA1E618A15` |
| `student_bert_4l384_int8.pt` | 15,851,841 B (15.12 MiB) | 存在，MD5 `D1A86278F7F033945528FF2E6E9CDD32` |
| `bert_multitask_classifier_model.pt` | ~390 MB | **缺失** |
| `cache/` 整个目录 | — | **不存在** |

`eval_result.txt` 第 166-173 行的备注明确写「model/ 目录只保留」这三个文件；
README 第 55-63 行也把它们列为目录应有内容。但实测 `model/` 只有 2 个，
`cache/`（`teacher_soft_labels.npz` 0.67 MB、`teacher_cls_hidden.npz` 237 MB、
`eval_result_baseline_backup.txt`）全部不存在。全盘搜索 `*multitask*`、`pytorch_model.bin`、
`*.safetensors` 均无结果。

**影响**

- `cache_teacher_logits.py` 无法运行 → 蒸馏链路的第一步就断了
- `distill_train.py --teacher-ref` 无法运行 → 教师基准无法复现
- `bert_eval_on_test.py` 无法运行 → test 评估无法复现
- `summarize_results.py` 的「基线」行会**静默消失**（`base_file` 不存在，代码里是 `if baseline:` 兜着）

**原因**：这三个大文件在 README 里都标注「不入库」（390 MB / 237 MB 确实不该进 git），
但仓库里**没有任何获取或重新生成的脚本/说明**，导致「不入库」= 「永久丢失」。
`cache/*.npz` 可由 `cache_teacher_logits.py --force` 重建，但它又依赖缺失的教师权重。

**修复建议**

1. 补一份「产物获取说明」：要么把教师权重放到 Release / 网盘并在 README 给出地址，
   要么明确写出「重训教师」的完整命令链。
2. 给 `eval_result.txt` 的备注段补一句：这些文件不在仓库内，以及如何获得。
3. 建议把 `cache/` 的生成做成一条幂等命令，并在 README 里作为「步骤 0」列出。

> **补充说明**：权重缺失是「按设计不入库」，不算代码 bug。但审查目标包含
> 「文件是否完整」，而当前状态的客观后果是：**这份副本无法运行、无法复现、无法继续迭代**。

---

### P1 —— 高价值：推理性能存在 4~9 倍可回收空间

#### 问题 4：固定 padding 到 256，浪费 76.1% 的计算

**证据**

val 集真实 token 长度分布（未截断）：

| 指标 | 中位数 | 均值 | P90 | 最大 |
|---|---|---|---|---|
| token 数 | 36 | **61.3** | 137 | 1356 |

而 `dataloader_utils.my_collate_fn:60-66` 和 `predict._forward:156-162` 都用
`padding='max_length', max_length=256` → **每条样本都按 256 个位置做完整前向**，
有效 token 只占 61.3 / 256 = 23.9%，即 **76.1% 的算力花在 padding 上**。

实测（val 前 1024 条，batch=64，2 次 warmup + 3 次取最小值）：

| 方案 | 分词 ms/条 | 前向 ms/条 | 合计 ms/条 | 相对现状 |
|---|---|---|---|---|
| fp32 固定 256（现状） | 0.28 | 20.49 | 20.76 | 1.00x |
| fp32 动态 padding | 0.25 | 19.30 | 19.55 | 1.06x |
| **fp32 动态 + 按长度分桶** | 0.08 | **2.07** | **2.15** | **9.64x** |
| int8 固定 256（现状） | 0.21 | 5.13 | 5.34 | — |
| **int8 动态 + 按长度分桶** | 0.08 | **1.06** | **1.15** | **4.66x** |
| int8 + 分桶，相对「现状 fp32」 | — | — | — | **18.13x** |

**为什么「动态 padding」单独用几乎没用，必须配「按长度分桶」**

`padding='longest'` 只按**当前 batch 内最长样本**对齐。随机组成的 batch 里只要混进一条
长评论，整批都要按长序列算，收益被拉平（实测只快 6%）。
把同一批内的样本按长度排序后再切 batch，各批长度才能贴近自身上限，收益才释放出来。

**内存/显存同样受益**

以 batch=64、6 头、fp32 为例，单个注意力分数矩阵（`B × heads × L × L`）：

| 序列长度 | 计算式 | 张量规模 |
|---|---|---|
| 固定 256 | 64 × 6 × 256² × 4 B | **96.00 MiB** |
| 分桶后（L≈61，即 val 均值） | 64 × 6 × 61² × 4 B | 5.45 MiB（**−94.3%**） |
| 同一批短文本（L=16） | 64 × 6 × 16² × 4 B | 0.375 MiB（**−99.6%**） |

实测同一批短文本的输入张量：从 `(64, 256)` = 384 KB 降到 `(64, 16)` = 24 KB，**16 倍**。
注意力矩阵是 BERT 里最大的单个中间张量，这一项直接决定峰值内存的量级。

**修复建议**

改动集中在 `dataloader_utils.my_collate_fn` 与 `predict._forward` 两处，**零精度风险**
（`attention_mask` 已正确屏蔽 padding，模型看到的有效输入完全相同）：

```python
# 1) collate_fn 改为按批内最长对齐
batch_texts_tensor = config.bert_tokenizer(
    texts, max_length=config.max_len, padding='longest',   # ← 由 'max_length' 改
    truncation=True, return_tensors='pt')

# 2) 评估/训练侧：按 token 长度排序后再切 batch（指标不受影响）
```

**`predict.py` 的注意点（重要）**：`predict_batch` 必须**按原始顺序返回结果**，
所以不能原地排序列表，要用索引重排后再散射回原顺序，例如：

```python
order = sorted(range(len(texts)), key=lambda i: lengths[i])
# ... 按 order 分块前向 ...
# 结果按 order 散射回原下标，再按 0..n-1 输出
```

**一个反直觉但有价值的发现**：量化收益会随序列变短而衰减 ——
固定 256 时 int8 比 fp32 快 **3.99x**，分桶到真实长度后只剩 **1.96x**。
原因是序列变短后，矩阵乘不再是瓶颈，**Embedding 查表、LayerNorm、GELU、softmax
以及 Python/算子派发开销**成了主导，而这些在 PyTorch 动态量化里仍是 fp32。
**结论：先做 padding 优化，再想靠量化挤性能，性价比会越来越低。**

---

#### 问题 5：词表 69.4% 的行从未被用到，这是目前最大的结构冗余

**证据**

对全部 62,600 条数据（train + val + test）做未截断分词，统计 token id 覆盖：

| 项目 | 数值 |
|---|---|
| 嵌入表行数 | 21,128 |
| 实际被用到的 token | **6,458（30.57%）** |
| 从未被用到的行 | **14,670（69.43%）** |
| 语料总 token 数 | 3,699,672 |

**收益测算**

| 指标 | 现状 | 词表裁剪后 | 节省 |
|---|---|---|---|
| 词嵌入参数 | 8,113,152（占 52.1%） | 2,479,872 | −5.37 M |
| int8 体积 | 15.12 MiB | **约 9.75 MiB** | −35.5% |
| fp32 体积 | 59.39 MiB | 约 37.90 MiB | −36.2% |

**为什么这件事比看起来便宜（关键优势）**

教师软标签缓存里存的是**输出 logits 和 [CLS] 隐状态**，与词表无关。
所以裁剪词表**不需要重跑 `cache_teacher_logits.py`（约 8 分钟）**，
只需改词表映射后重跑 `distill_train.py`（约 27 分钟）。而且学生本来就是随机初始化从零训练的，
**没有需要搬运的预训练词向量，重训没有任何额外代价** —— 这条路径在本项目里异常顺畅。

**前提与风险**

- 需要重新映射 token id，并同步更新 `vocab.txt` / `tokenizer.json` / `config.json` 的 `vocab_size`
- 必须留 OOV 余量：建议保留「实际用到的 token + 全部单字 + ASCII + 特殊符」，目标 8,000~10,000 行
- 生产环境若出现训练集未见过的字，会被映射成 `[UNK]`，这是精度换体积的取舍，需要业务侧确认

---

#### 问题 6：量化加速比的数字与测量口径（记录值本身可信，但缺少环境标注）

**证据**

`eval_result.txt` 记录：`25.94 → 18.55 ms/条（1.40× 加速）`。
本机在**项目自带的评估口径**下实测（`test_predict.py` 第 6 项，跑完整 9390 条 val）：

| 测量口径 | fp32 | int8 | 加速比 |
|---|---|---|---|
| 项目自带 `test_predict.py`，全量 val，固定 256，batch=64 | 186.9 s / 19.91 ms 条 | 104.8 s / 11.17 ms 条 | **1.78x** |
| 本次审查独立脚本，全量 val，固定 256，batch=64 | 165.2 s / 17.59 ms 条 | 78.5 s / 8.36 ms 条 | 2.10x |
| 本次审查独立脚本，前向-only，1024 条，固定 256 | 20.49 ms/条 | 5.13 ms/条 | 3.99x |
| 同上，改为动态 padding + 分桶后 | 2.07 ms/条 | 1.06 ms/条 | 1.96x |

**修正说明**：本报告初版曾据此认为「记录的 1.40x 偏保守」。用项目自带口径复测后（1.78x），
更准确的结论是：**这个数字与机器/线程/测量口径强相关，实测区间大致在 1.4x ~ 2.1x**，
记录值处于合理范围内，只是**没有标注测试环境**（CPU 型号 / 线程数 / padding 策略），
因而无法作为通用选型依据引用。

**建议**：报告里给性能数字时同时标注环境（CPU 型号 + 线程数 + padding 策略 + 批大小）。

| 测量口径 | fp32 | int8 | 加速比 |
|---|---|---|---|
| 全量 val（9390 条，含分词，batch=64，固定 256） | 165.2 s / 17.59 ms 条 | 78.5 s / 8.36 ms 条 | **2.10x** |
| 前向-only（1024 条，batch=64，固定 256） | 20.49 ms/条 | 5.13 ms/条 | **3.99x** |
| 前向-only（1024 条，batch=64，分桶后） | 2.07 ms/条 | 1.06 ms/条 | 1.96x |

**建议**：报告里给数字时**同时标注测试环境**（CPU 型号 + 线程数 + padding 策略）。

---

### P2 —— 中等问题

#### 问题 7：学生模型从未在 test 集上评估（流程缺口）

> ⚠️ **本节的初版推断「对外数字约高估 2 个点」已被实测推翻，见下方更正。**
> 标题保留，因为**流程缺口本身是真实的**：仓库里确实没有任何学生的 test 集记录。

**证据**

- 本模块 `eval_result.txt` 的全部条目（教师、A/B/C、fp32、int8）都是**验证集**数字
- `bert_eval_on_test.py` 加载的是 `config.bert_classifier_model_save_path`，
  也就是**教师**权重，不是学生
- 全模块搜索 `test_dataloader`：只有 `bert_eval_on_test.py`（教师）和
  `bert_classifier_model.py` 的 `__main__` 演示用到

**教师确实存在这个落差 —— 但学生没有（这一点我最初推断错了，已实测更正）**

`models/bert/eval_result.txt` 里记录了教师在 test 集上的成绩：

| 教师 | 商品大类 F1 | 情感 F1 | 综合 F1 |
|---|---|---|---|
| 验证集（本模块 `eval_result.txt` 【0】） | 0.9208 | 0.9607 | **0.9407** |
| **测试集**（`models/bert/eval_result.txt`） | 0.8949 | 0.9413 | **0.9181** |
| 落差 | −0.0259 | −0.0194 | **−0.0226** |

教师在同一套数据上**验证集比测试集乐观约 2.3 个点**。

> ⚠️ **更正说明**：本报告初版据此推断「学生真实水平约在 0.88~0.89」。**该推断是错的。**
> 我把学生模型在 test 集上真跑了一遍（`docs/model_audit/test_set_eval.py`），结果如下：

| 学生 | 商品大类 F1 | 情感 F1 | 综合 F1 |
|---|---|---|---|
| **验证集** fp32 | 0.8943 | 0.9202 | 0.9072 |
| **测试集** fp32 | 0.8928 | 0.9236 | **0.9082** |
| **验证集** int8 | 0.8944 | 0.9203 | 0.9074 |
| **测试集** int8 | 0.8941 | 0.9232 | **0.9086** |

**学生几乎没有 val→test 落差**（fp32 甚至 +0.0010）。所以：

- 对外数字 `0.9072` 作为验证集成绩是**诚实的**，测试集实测 **0.9082**，两者一致
- 教师的 2.3 点落差**没有传递给学生** —— 学生泛化得比教师更稳
- **更重要的发现**：在**独立测试集**上，教师 0.9181 vs 学生 0.9082 → **保留率 98.92%**
  （int8 为 98.96%），远高于 README 第 8.1 节按验证集算出的 96.44%

**这才是应该对外讲的数字**：学生用 **1/6.6 的参数**（15.56 M vs 102.27 M）和
**1/25.8 的体积**（15.12 MB vs 390.22 MB），在独立测试集上保留了教师 **98.9%** 的性能，
且大类 F1（0.8941 vs 教师的 0.8949）几乎持平。

**仍建议做的流程性修补**（成本 3 分钟）：把 `bert_eval_on_test.py` 复制一份改为评估学生
（加载 `config.student_model_path` 与 `config.quantized_model_path`），把这两个 test 数字
正式写进 `eval_result.txt`。目前这两个数字**只存在于本次审查的临时脚本里，仓库中没有记录**。

---

#### 问题 8：62 MB / 15 MB 二进制权重直接进 git，无 Git LFS

**证据**

| 项目 | 数值 |
|---|---|
| `.git` 目录体积 | **443 MB** |
| `.gitattributes` | **不存在** |
| `git lfs ls-files` | 无输出（未使用 LFS） |
| 被 git 跟踪的大文件 | `models/rf/model/rf_model.pkl` **257 MB**、`rf_model_sentiment.pkl` **181 MB**、`ft_joint_default.bin` 64 MB、`student_bert_4l384.pt` **62 MB**、`student_bert_4l384_int8.pt` **15 MB** |

**影响**：任何人 clone 一次仓库都要拉 443 MB，且这些二进制一旦进历史就很难瘦身。
本模块贡献了 77 MB。远端是 Gitee，大文件推送也容易触发限制。

**修复建议**

```
# 1) 配置 LFS（新增 .gitattributes）
*.pt    filter=lfs diff=lfs merge=lfs -text
*.pkl   filter=lfs diff=lfs merge=lfs -text
*.bin   filter=lfs diff=lfs merge=lfs -text
```

或更彻底：模型产物不进仓库，改为 CI 产物 / Release 附件，
仓库里只留 `MODEL_CARD.md`（含 MD5 校验值）。本模块已有现成可用的校验值
（`4CCC792676306C727DEB41BA1E618A15` / `D1A86278F7F033945528FF2E6E9CDD32`），直接写进文档即可。

---

#### 问题 9：两套并行的 `max_len` 配置，是潜在的静默不一致源

**证据**

| 使用位置 | 读取的配置项 | 当前值 |
|---|---|---|
| `dataloader_utils.my_collate_fn:62` | `config.max_len` | 256 |
| `distill_data._encode:81` | `config.distill_max_len` | 256 |
| `predict._forward:158` | `config.distill_max_len` | 256 |
| `bert_config.py:118` 注释 | 「必须与教师一致，否则软标签分布不可比」 | — |

两个值现在恰好都是 256，所以没有暴露问题。但它们是**两个独立可改的旋钮**：
调了 `max_len` 忘了调 `distill_max_len`，会导致**教师软标签与学生的输入长度不一致**，
而这类错误不会报错，只会让蒸馏悄悄退化成「学生学了一个错位的分布」。

**修复建议**：合并为单一配置项（保留 `distill_max_len` 作为唯一真源，
或让 `max_len` 变成 `distill_max_len` 的别名），并加一句断言：

```python
assert self.max_len == self.distill_max_len, \
    "教师软标签缓存按 distill_max_len 生成，学生输入长度必须一致"
```

---

#### 问题 10：教师类共享全局编码器实例，两个实例会互相干扰

**证据**

`bert_classifier_model.py:15`：`self.bert = config.bert_model`，而 `config.bert_model`
来自 `Config._model_cache` 的全局缓存。实测：

```
t1.bert is t2.bert  -> True   （两个教师实例共用同一份编码器）
t1.bert is 全局基座 -> True
```

**影响**：当前流程里每次只创建一个教师实例，所以没出事。但这是一个**静默的别名（aliasing）陷阱**：
一旦有人同时建两个 `MyBertMultiTaskClassifier()`（比如想对比两个检查点），
给其中一个 `load_state_dict`，另一个的权重会被**同步改掉**，且不会报任何错。
`quantize.py` 恰好因为 `quantize_dynamic` 默认 `deepcopy` 才躲过同类问题（见问题 11）。

**修复建议**：`__init__` 里改为 `self.bert = copy.deepcopy(config.bert_model)`，
或直接 `transformers.BertModel(config.bert_config)` 新建一份。

---

### P3 —— 低风险 / 文档准确度

| # | 问题 | 证据（文件:行） | 影响 | 建议 |
|---|---|---|---|---|
| 11 | 注释说 `quantize_dynamic` 是「就地替换子模块并返回同一个对象」，**实测不符** | `quantize.py:11`；实测 `q is m → False`，签名 `inplace=False` 为默认 | 量化时会**多占一份 fp32 内存**（本模型 62 MB，可忽略；放在 390 MB 教师上就不是小数目）。当前流程先测 fp32 再量化，所以没有正确性影响 | 把注释改为「默认深拷贝，返回新对象；如需省内存可传 `inplace=True`」 |
| 12 | `vocab.txt` 全部 21,128 行以 `\r` 结尾（CRLF），来自 ModelScope 下载 | `bert-base-chinese/vocab.txt`；实测以 `\r` 结尾行数 = 21128/21129 | **当前无影响**：tokenizer 是 fast 版（`is_fast=True`），实际读 `tokenizer.json`，实测分词 0 个 `[UNK]`。但若日后 `use_fast=False` 或用别的工具直接读 `vocab.txt`，会得到带 `\r` 的 token | 统一转成 LF |
| 13 | 位置嵌入 512 行，但 `max_len=256`，后一半永远取不到 | `bert-base-chinese/config.json` `max_position_embeddings=512`；`distill_max_len=256` | 白占 98,304 参数（int8 约 0.094 MiB）。数值上无害 | 学生侧可设 `max_position_embeddings=256`（学生与学生自己的输入绑定，不影响教师软标签） |
| 14 | `bert_eval_on_test.py` 未用 `weights_only=True` | `bert_eval_on_test.py:22` | 与其余模块（都加了 `weights_only=True`）不一致；`torch.load` 不加该参数会执行任意 pickle 代码 | 统一加上 |
| 15 | `bert_config.py:195` 打开 `class.txt` 未关闭句柄 | `bert_config.py:195` | 文件句柄泄漏（`ResourceWarning`），单次影响可忽略 | 用 `with open(...) as f:` |
| 16 | 推理入口为拿一个函数间接引入 pandas / sklearn / tqdm | `predict.py:119` `from quantize import quantize_model` → `quantize.py` 导入 `dataloader_utils`(pandas) + `report_utils`(sklearn) | 服务端启动多加载三个重依赖，拖慢冷启动 | 把 `quantize_model()` 抽到一个不依赖数据/评估的轻量模块 |
| 17 | 路径拼接混用 `\` 与 `/` | `bert_config.py:42` `_PROJECT_ROOT + '/'`；实测输出 `D:\...\ai8_-project1/models/bert-base-chinese` | 可用但观感差，跨平台判断易踩坑 | 统一用 `os.path.join` / `pathlib` |
| 18 | `eval_result.txt` 记录了另一台机器的绝对路径 | 该文件第 17/31/50/69/88/103 行：`C:\Users\29011\PycharmProjects\...` | 与「不再硬编码绝对路径」的改造方向矛盾；也泄露他人目录结构 | 报告里改为相对路径 |
| 19 | `summarize_results.py` 依赖 `cache/eval_result_baseline_backup.txt`，该文件当前不存在 | `summarize_results.py:78-81` | 「基线」那一行会**静默消失**，不报错 | 缺失时打印提示 |
| 20 | `format_metrics_block` 的 `index` 参数类型混用 | `distill_train.py:262` 传 `0`（int）；同文件 270 行传 `args.exp`（str） | 输出 `【0】` 与 `【A】` 风格不一致 | 统一转成字符串 |

---

## 三、验证通过项（这些地方做得对，不要误改）

这一节是**实测确认无问题**的结论，避免后续为了「优化」反而改坏了。

### 3.1 权重数值干净

对两个 `.pt` 逐张量扫描 15,560,457 个数值：

- 无 NaN、无 Inf、无全零张量
- 所有 LayerNorm 的 `gamma` 都在 0.90~1.06 之间（std ≈ 0.015），**没有退化**
- 注意力 `key.bias` 量级 ~1e-4 而 `key.weight` 量级 ~0.025，这是 BERT 的正常特征，不是异常

### 3.2 文件结构完整，无丢权重

| 项目 | fp32 | int8 |
|---|---|---|
| `load_state_dict(strict=True)` | **通过** | **通过** |
| 参数量 | 15,560,457（与记录一致） | — |
| 元素总数 | 15,560,457 | 15,560,511 |
| 差值 | — | **+54 = 27 个 scale + 27 个 zero_point**（量化元数据） |

`strict=True` 能过，意味着**权重与代码结构严丝合缝，无缺失、无多余键**。
int8 的 54 个额外标量是量化必需的元数据，**验证集权重零丢失**。

> 补充：`int8` 状态字典里的 Linear 权重被包在 `_packed_params._packed_params` 元组里
> （不是直接张量），朴素统计会漏算 7,243,017 个元素，容易误判成「权重丢了」。
> 本次已解包核对，确认无丢失。

### 3.3 量化精度损失极小 —— 这才是真正的亮点

逐个反量化后与 fp32 对比（27 个 Linear）：

| 模块 | 相对误差 | 余弦相似度 |
|---|---|---|
| 最差（`layer.0.output.dense`） | 1.537% | 0.99986 |
| 平均 | **1.163%** | — |
| 最好（`cat_linear`） | 0.536% | 0.99999 |
| 词嵌入整体 | **0.666%** | — |

**结论：量化对权重本身几乎没有破坏。** 这也解释了为什么 int8 的 F1 与 fp32 完全一致。

### 3.4 关于 `scale=1.0` —— 已核实是无害的，不是 bug

int8 词嵌入有 1 行（21,128 行中）的量化 `scale` 异常地等于 1.0（其余行中位数 4.7e-4）。
逐行核对：

```
scale == 1.0 的行数: 1 / 21128
  这些行在 fp32 中的最大绝对值   : 全部 = 0  → True
  在 fp32 非零、但反量化后变全 0 的行: 0 行
```

也就是说，**这一行在 fp32 里本来就是全 0**（一个未被使用的词表行），
PyTorch 观察器对「min==max」的退化行返回 `scale=1.0` 是标准行为。
**没有任何一行是被量化破坏掉的**，无需处理。

### 3.5 体积压缩已达 int8 理论上限

int8 文件 15.12 MiB 的构成：

| 部分 | 体积 | 占比 |
|---|---|---|
| 词嵌入 (21128×384) | 7.898 MiB | 52.4% |
| Linear 打包（27 个） | 6.948 MiB | 46.1% |
| 位置/类型嵌入 | 0.192 MiB | 1.3% |
| LayerNorm（保持 fp32） | 0.026 MiB | 0.2% |
| pickle 元数据开销 | 0.050 MiB | 0.3% |
| **合计** | **15.12 MiB** | — |

59.39 → 15.12 MiB = **3.93x**，而 int8 的理论上限就是 4x。
**压缩这条路已经走到头了**，剩下的空间只能靠减少参数量（见问题 5）。

### 3.6 指标 100% 复现

本机重建模型（自己按 `config.json` + 学生规格构造同构网络，绕开崩溃的 `Config()`），
在真实 `val.csv`（9390 条）上跑完整评估：

| 模型 | 大类 F1 | 情感 F1 | 综合 F1 |
|---|---|---|---|
| fp32 实测 | 0.8943 | 0.9202 | **0.9072** |
| fp32 记录值 | 0.8943 | 0.9202 | 0.9072 |
| int8 实测 | 0.8944 | 0.9203 | **0.9074** |
| int8 记录值 | 0.8943 | 0.9204 | 0.9074 |

**逐项吻合到小数点后 4 位。** `eval_result.txt` 里的模型指标是可信的、可复现的。

**并且在测试集上同样站得住**（本次补齐，详见问题 7）：

| 模型 | 验证集 综合 F1 | **测试集 综合 F1** | 差值 |
|---|---|---|---|
| fp32 学生 | 0.9072 | **0.9082** | +0.0010 |
| int8 学生 | 0.9074 | **0.9086** | +0.0012 |
| （教师，作对照） | 0.9407 | 0.9181 | −0.0226 |

学生的验证集与测试集**基本持平**，不存在教师那样 2.3 个点的落差。
在独立测试集上学生对教师的保留率为 **98.92%（fp32）/ 98.96%（int8）**。

另外，项目自带的 `test_predict.py` 测试套件本次实测 **31 项全部通过、0 失败**，
其中第 6 项「端到端跑完整验证集复现 `eval_result.txt` 指标」成功复现。

### 3.7 蒸馏配置合理

| 检查项 | 结论 |
|---|---|
| 温度 T=2.0 / α=0.7 | 合理区间；T² 只乘在软标签项上（`distill_train.py:120`），**正确** |
| KL 方向 | `F.kl_div(log_softmax(学生/T), log_softmax(教师/T), log_target=True)` 展开即 `KL(教师‖学生)`，**正确** |
| 层映射 `(3,6,9,12)` | HF `hidden_states` 索引 0=embedding 输出、i=第 i 层输出，故 12 = 末层；学生取 `hidden_states[k+1]` (k=0..3) = 第 1..4 层输出。**索引正确，无 off-by-one** |
| 学生初始化 | 随机初始化 + 预训练级 lr 3e-4（`distill_train.py` 注释里记录了从 5e-5 修正的过程），**正确** |
| 学习率调度 | OneCycleLR，`warmup_ratio=0.1`，`div_factor=10`，`final_div_factor=100`，逐批 `step()`，**正确** |
| 梯度裁剪 | `clip_grad_norm_(学生+投影层, 1.0)`，**正确**（投影层也被裁剪） |
| 最优模型选择 | 逐轮验证集 `avg_f1` 比较后保存，**正确** |
| 数据泄漏 | 教师软标签只缓存 `train.csv`；模型选择用 `val`；`test` 未参与。**无泄漏** |
| 量化 qconfig | `Linear: default_dynamic_qconfig` + `Embedding: float_qparams_weight_only_qconfig`，**正确且必要**（Embedding 占 52%，不量化则压缩收益减半） |
| 量化引擎 | `onednn`（本机实测 `supported_engines == ['onednn']`），**正确** |
| 量化加载顺序 | 实测两种方式都与文档一致；README 第九节 #1~#5 的踩坑记录**准确** |

### 3.8 代码质量基线良好

- 14 个 `.py` 文件全部通过 `py_compile`
- 每个模块的文件头都有**为什么这么做**的注释，且记录了踩坑与修正过程（README 第九节 15 条）
- `test_predict.py` 有 31 项断言，包含「端到端跑完整验证集复现指标」这种硬核校验
- 模型选择、量化前后对比、报告落盘都有统一封装（`report_utils.py`），格式一致可直接横向比较

---

## 四、优化建议（按性价比排序）

| 优先级 | 措施 | 预期收益 | 成本 | 风险 | 参考 |
|---|---|---|---|---|---|
| ⭐⭐⭐ | **补齐基座 / 教师 / 缓存文件**，或补上获取说明 | 从「完全跑不起来」到「可运行」 | 低 | 无 | 问题 1、3 |
| ⭐⭐⭐ | **动态 padding + 按长度分桶** | fp32 **9.64x**；int8 与现状 fp32 相比 **18.13x** | 改约 20 行，2 小时 | **零精度风险**（注意力掩码已正确屏蔽） | 问题 4 |
| ⭐⭐⭐ | **词表裁剪到实际用到的规模** | int8 15.12 → **9.75 MiB（−35.5%）**；fp32 59.39 → 37.9 MiB | 重跑蒸馏 27 分钟（**教师缓存可复用，无需重跑**） | 中（OOV 需留余量，需业务确认） | 问题 5 |
| ⭐⭐⭐ | **修 P0-2 单例半初始化** | 报错信息回到真因 | 改 3 行 | 无 | 问题 2 |
| ⭐⭐ | **补学生的 test 集评估**（数字本次已测出，只需补进仓库） | 让 98.92% 保留率这个更强的结论有仓库记录 | 复制脚本改 2 行，3 分钟 | 无 | 问题 7 |
| ⭐⭐ | **蒸馏轮数提到 12~16**（README 8.4 已建议） | 预计 +0.5~1.5 点 | 纯算力，约 15 分钟 | 低 | 问题 7 附录 |
| ⭐⭐ | **学生换 6L/512 或 4L/768**（宽度别砍半） | 预计 +1~3 点，体积约 115 / 60 MB | 重跑蒸馏 | 低 | — |
| ⭐⭐ | **配置 LFS 或把产物移出仓库** | clone 体积从 443 MB 降下来 | 低 | 需重写历史或只对新文件生效 | 问题 8 |
| ⭐ | **补 T / α 消融**（α=0 纯硬标签 vs α=1 纯软标签） | 自证软标签确实有效（目前只有理论依据） | 2 次蒸馏，约 54 分钟 | 无 | — |
| ⭐ | **试 `torchao` 的 int8/int4 weight-only 量化** | 可能进一步压到 ~4~8 MiB | 需新装依赖 | 中，**本机未安装 torchao，此条未经实测** | — |
| ⭐ | **统一 `max_len` 配置 + 加断言 / 修 P3 各项** | 消除静默不一致 | 低 | 无 | 问题 9~20 |

### 关于「静态量化 / QAT 不可用」这一结论的建议

README 第 5.7 节记录：「FX Graph Mode 静态量化 / QAT 在 HF BERT 上报
`TypeError: slice indices must be integers`，因此动态量化是唯一可行路径」。

这个**现象**是准确的（FX symbolic tracing 与 HF BERT 里依赖运行时数值的切片确实冲突），
但「唯一可行路径」这个**结论**略强。README 自己也提到官方建议迁移 `torchao` ——
而 `torchao` 的 `int8_weight_only` / `int4_weight_only` 走 **eager 模式**，
**不需要 FX tracing**，正好绕开这个限制。这条路的收益很直接：int4 有望把 15.12 MiB 压到 ~8 MiB。

> 需要说明的是：本机 **未安装 `torchao`**，这条建议**未经实测验证**，仅作为方向性参考。
> 若采纳，建议先用 `student_bert_4l384.pt` 做离线试验，确认精度无损再替换。

---

## 五、本次审查的覆盖范围与未覆盖项

### 已覆盖（均为实测，非推断）

- 目录文件盘点（含大小、MD5、git 跟踪状态、LFS 状态）
- 全部 14 个 `.py` 的语法编译与导入可行性
- 两个 `.pt` 的结构、dtype、数值分布、异常检测、逐模块量化误差
- int8 权重零丢失核对（解包 27 个 `_packed_params`）
- 真实 `val.csv`（9390 条）端到端指标复现
- 分词成本 / 前向成本 / 5 种输入长度策略的性能基准
- 词表覆盖率（全部 62,600 条数据）
- 分词器行为（fast/slow、UNK 率）
- 蒸馏超参与层映射的正确性核对
- 教师 val→test 落差（用于推断学生真实水平）

### 未覆盖（客观限制）

| 项目 | 原因 |
|---|---|
| 学生的 test 集实测指标 | **已补测完成**（0.9082 / 0.9086），见问题 7；脚本 `docs/model_audit/test_set_eval.py` |
| 基座权重缺失问题 | **已修复**（已下载 `model.safetensors` 至 `bert-base-chinese/`） |
| 重跑蒸馏 / 教师缓存 / 教师 test 评估 | 教师权重（~390 MB）仍缺失，无法执行 |
| 学生 6L/512 等更大结构的收益 | 需要重新训练，不在「审查」范围 |
| `torchao` / ONNX Runtime / OpenVINO 的加速效果 | 相关库未安装，未做实测 |
| 进程峰值 RSS 精确值 | 沙箱环境下 `GetProcessMemoryInfo` 返回 0，**未取得可信读数**，故本报告只给可精确计算的张量规模（如注意力矩阵 96 MiB → 更小），不编造内存数字 |
| 多线程 / 并发下的吞吐 | 本次为单进程基准 |

---

## 六、可复现的检查工具

本次审查使用的诊断脚本已保存在：

```
docs/model_audit/
├── inspect_pt.py          # 权重结构 / dtype / 数值分布 / 异常检测（纯 torch，不需 transformers）
├── deep_quant.py          # 解包量化权重 + 逐模块量化误差 + 词嵌入逐行检查
├── rebuild_bench.py       # 重建模型 + 真实 val 复现指标 + 5 种输入策略基准
├── bench2.py              # 分词/前向拆分计时（warmup + 多次取最小）
└── check_config_boot.py   # 验证基座权重缺失导致的启动失败
```

运行方式（以 `PYTHON_ML` 为例）：

```powershell
$py = "D:\conda_envs\PYTHON_ML\python.exe"
& $py -u docs\model_audit\inspect_pt.py
```

这些脚本**不修改项目任何文件**，可以随时重跑复核。

---

## 附录：关键数据速查

```
文件清单
  bert-base-chinese/     ✅ 已补齐权重：model.safetensors 411,553,788 B (392.49 MiB)
                            （原缺，2026-09-22 上午从 hf-mirror 下载；已被 .gitignore 忽略）
  model/student_bert_4l384.pt        62,273,746 B  (59.39 MiB)  MD5 4CCC792676306C727DEB41BA1E618A15
  model/student_bert_4l384_int8.pt   15,851,841 B  (15.12 MiB)  MD5 D1A86278F7F033945528FF2E6E9CDD32
  model/bert_multitask_classifier_model.pt  ❌ 仍缺失（~390 MB）
  cache/                                    ❌ 仍缺失（整个目录）

模型规格
  学生 4 层 / 384 隐层 / 6 头 / FFN 1536 / 词表 21128 / max_position 512
  参数量 15,560,457（词嵌入占 52.1%）
  教师 12 层 / 768 隐层 / 12 头 / FFN 3072，参数量 102,274,569

量化
  方式   动态量化 (torch.ao.quantization)，引擎 onednn
  范围   nn.Linear (qint8) + nn.Embedding (float_qparams, quint8)
  误差   Linear 0.536%~1.537% / 余弦 >0.9998；词嵌入 0.666%
  体积   59.39 → 15.12 MiB = 3.93x（理论 4x）
  权重   零丢失（15,560,511 = 15,560,457 + 54 个 scale/zp）

指标（本机复现）
  【验证集 9390 条】
    fp32  大类 0.8943  情感 0.9202  综合 0.9072   ← 与 eval_result.txt 一致
    int8  大类 0.8944  情感 0.9203  综合 0.9074   ← 与 eval_result.txt 一致
  【测试集 9390 条】★ 本次补齐
    fp32  大类 0.8928  情感 0.9236  综合 0.9082
    int8  大类 0.8941  情感 0.9232  综合 0.9086
  教师（对照） val 综合 0.9407  →  test 综合 0.9181（落差 2.26 点）
  学生对教师保留率（test 集）fp32 98.92% / int8 98.96%

项目自带测试
  test_predict.py  31 项全部通过 / 0 失败（含端到端复现 val 指标）

性能（val 前 1024 条，batch=64，16 线程，取 3 次最小）
  分词      0.08~0.28 ms/条（占比 1.3%~3.9%，非瓶颈）
  fp32 前向 固定256 20.49 → 分桶 2.07 ms/条   （9.64x）
  int8 前向 固定256  5.13 → 分桶 1.06 ms/条   （4.66x）
  int8 加速比 固定256 3.99x → 分桶后 1.96x（收益随序列变短而衰减）

数据（val 集）
  token 长度 中位 36 / 均值 61.3 / P90 137 → padding 到 256 浪费 76.1%
  词表覆盖   21,128 行中仅 6,458 被用到（30.57%），14,670 行是死重
```

---

*报告生成：2026-09-22 | 全部数字均为本机实测，未实测的项目已在第五节明确标注*

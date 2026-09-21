# 电商评论智能分类系统

## 📖 项目简介

基于BERT的电商评论智能分类系统，通过分析用户评论内容，自动判断评论所属的商品大类，并预测情感倾向。

## 🎯 项目任务

### 主任务：商品大类7分类

| 一级大类（7个） | 包含品类 | 原始样本量 | 清洗后样本量 |
|---|---|---|---|
| 数码电子 | 手机、平板、电脑 | 16,315 | 16,241 |
| 食品饮料 | 乳制品、水果 | 12,033 | 12,019 |
| 个护美妆 | 洗发水 | 10,000 | 9,997 |
| 服饰鞋包 | 衣服 | 10,000 | 9,994 |
| 本地生活 | 酒店 | 10,000 | 9,955 |
| 图书文娱 | 书籍 | 3,851 | 3,850 |
| 家用电器 | 热水器 | 575 | 544 |
| **合计** | **10 个原始类别** | **62,774** | **62,600** |

> 清洗共去除 174 条（1 条空评论、153 条清洗后不足 5 字、2 条纯符号、18 条重复评论）。另有 80 行评论中的手机号、邮箱、订单号等个人信息已遮盖为占位符（如 `[手机号]`）。**模型训练使用清洗后的 62,600 条**，按 70/15/15 分层划分为 train 43,820 / val 9,390 / test 9,390。

### 辅助任务：情感正负二分类
判断评论是正面评价还是负面评价。`label` 列取值为 **0 = 负面，1 = 正面**。

## 🛠️ 技术栈

- **编程语言**：Python 3.9+
- **深度学习框架**：PyTorch
- **预训练模型**：BERT-base-chinese
- **传统机器学习**：Scikit-learn（随机森林）
- **快速文本分类**：FastText
- **Web部署**：FastAPI + HTML
- **版本管理**：Git + Gitee

## 📁 目录结构

```
ai8_-project1/
├── data/
│   ├── raw/                    # 原始数据（online_shopping_10_cats.csv）
│   ├── processed/
│   │   └── final_data/         # 最终数据（7大类 + 情感）
│   └── figures/final_data/    # EDA可视化图表（9张，由第3步生成）
├── src/                       # 代码目录
│   ├── config.py                   # 配置入口（模型组与后续脚本使用）
│   ├── data_preprocess.py          # 数据清洗脚本（第1步）
│   ├── generate_fasttext_data.py   # FastText格式数据生成（第2步）
│   ├── eda_analysis.py             # EDA分析脚本（第3步）
│   └── data_audit.py               # 数据质量审计（只读，可随时复核）
├── models/                    # 训练好的模型文件
├── docs/                      # 项目文档
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明
```

## 👥 团队分工

| 小组 | 成员 | 职责 |
|---|---|---|
| 数据组 | 蔡隆宸、李欣祥 | 数据清洗、EDA分析、特征工程 |
| 模型组 | 待补充 | 随机森林、FastText、BERT模型训练与评估 |
| 优化部署组 | 待补充 | 模型蒸馏/量化、FastAPI部署、Web页面 |
| 文档组 | 待补充 | 项目文档、答辩PPT、答辩排练 |

## 📊 项目进度

- [x] 数据清洗与预处理
- [x] 7分类体系构建
- [x] EDA探索性分析（9张图表，见 `data/figures/final_data/`）
- [ ] 特征工程（分词 + TF-IDF）
- [ ] 基线模型训练（随机森林 + FastText）
- [ ] BERT模型训练与调优
- [ ] 模型对比与评估
- [ ] 模型优化（蒸馏/量化）
- [ ] FastAPI + Web部署
- [ ] 答辩PPT制作与排练

## 🚀 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### 数据预处理

三个脚本按顺序执行即可。脚本按自身位置定位数据，**在任何目录下运行都可以**：

```bash
python src/data_preprocess.py            # 第1步：映射大类 + 清洗 + 分层划分
python src/generate_fasttext_data.py     # 第2步：生成 FastText 格式数据
python src/eda_analysis.py               # 第3步：生成 EDA 图表
```

> EDA 会自动探测本机的中文字体（Windows/macOS/Linux 字体名不同，仓库里不带字体文件）。找不到字体时**跳过图 9（词云）并打印提示，不会中断**，其余 8 张图正常生成；此时图中中文可能显示为方框，安装任一中文字体即可。

### 数据质量复核

```bash
python src/data_audit.py                 # 只读脚本，打印本文档引用的各项指标
```

### final_data 文件说明

| 文件 | 来源 | 说明 |
|---|---|---|
| `train.csv` / `val.csv` / `test.csv` | 第1步输出 | 列：`review,label,cat_l1`；按 **一级大类 + 情感** 组合分层，70/15/15 |
| `class.txt` | 第2步输出 | 7 个一级大类名，**行序即标签编号 0–6** |
| `train.txt` / `val.txt` / `test.txt` | 第2步输出 | FastText 格式 `文本<Tab>类别编号`（仅一级大类标签） |
| `stopwords.txt` | **手工维护的输入** | 非脚本生成，供 EDA 分词使用 |

> 注意：`class.txt` 的行序就是标签编号，**改动行的顺序会让 `train.txt` 等文件中所有标签的含义一起改变**。`stopwords.txt` 是人工维护的词表，不由任何脚本生成，删掉后需自行补回。原始数据的 10 个类别已在清洗阶段映射为 7 个大类，CSV 中不再保留原始类别列。
>
> 分层键同时包含类别与情感（共 7 × 2 = 14 个格子），因此 train / val / test 的正负样本比例一致（各类偏差 < 0.25pp），**在 val 上调出的结论可以直接迁移到 test**；若只按类别分层，各类别正负比例会在三份集合间漂移，最严重的达 10pp 以上。
>
> **编码约定**：CSV 为 UTF-8 **带 BOM**（便于 Excel 直接打开中文不乱码），TXT 为 UTF-8 **不带 BOM**；行尾均为 CRLF。读取 CSV 时请显式写 `encoding='utf-8-sig'`，否则 pandas 会把首列名读成 `\ufeffreview`；读取 TXT 用文本模式即可（若按二进制解析，注意行尾带 `\r`）。

### 读取 train.txt / val.txt / test.txt

这三个文件是 **TSV 格式**（每行 `文本<Tab>类别编号`），**不是 FastText 官方格式**（官方为 `__label__编号 文本`）。两点必须注意：

**① 用 pandas 读取时必须加 `quoting=csv.QUOTE_NONE`。** 有 4 条评论以 `"` 开头，默认引号规则会把它们当作带引号的字段并**合并相邻行**，train 会静默少读 487 行（43,820 → 43,333）而不报错：

```python
import csv
import pandas as pd

df = pd.read_csv('data/processed/final_data/train.txt', sep='\t', header=None,
                 names=['review', 'label'], quoting=csv.QUOTE_NONE, encoding='utf-8')
# 应读到 43,820 行；若只有 43,333 行，就是漏了 quoting=csv.QUOTE_NONE
```

**② 若要使用 `fasttext` 命令行或官方 API**，需先转成官方格式（类别号在前并加 `__label__` 前缀）：

```python
for split in ('train', 'val', 'test'):
    with open(f'data/processed/final_data/{split}.txt', encoding='utf-8') as fin, \
         open(f'data/processed/final_data/{split}_ft.txt', 'w', encoding='utf-8') as fout:
        for line in fin:
            text, label = line.rstrip('\n').split('\t')
            fout.write(f'__label__{label} {text}\n')
```

### 给模型组的配置与数据读取

`src/config.py` 是模型组与后续新脚本的统一入口：

```python
from config import TRAIN_CSV, VAL_CSV, TEST_CSV, CLASS_ORDER, SENTIMENT_MAP, MAX_LEN, load_csv
```

脚本放在 `src/` 下即可直接导入（无需任何额外处理）。若确有脚本要放在 `src/` 之外，先加两行把它纳入搜索路径：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```

直接运行 `python src/config.py` 可自检配置（打印路径、类别编号、各划分行数）。

**标签定义从 `class.txt` 派生，不要另抄一份。** `CLASS_ORDER` / `CLASS_TO_ID` 读的就是 `final_data/class.txt` 的行序 —— 它同时是 `train.txt` 等文件里类别编号的权威定义，因此不会与实际数据脱节。

**推荐用封装好的读取函数，两个坑已经封在函数里：**

| 函数 | 返回 | 说明 |
|---|---|---|
| `load_csv(split)` | DataFrame | 读 `train/val/test.csv`，自动处理 BOM（否则首列名会变成 `\ufeffreview`） |
| `load_fasttext_split(split)` | `(texts, labels)` | 读 `train/val/test.txt`，内部用 `QUOTE_NONE`（否则静默少读 487 行） |
| `load_stopwords()` | set | 停用词表 |

**数据路径**：`TRAIN_CSV` / `VAL_CSV` / `TEST_CSV` / `TRAIN_TXT` / `VAL_TXT` / `TEST_TXT`；需要按名字遍历用 `split_csv(split)` / `split_txt(split)`。另有 `DATA_DIR`、`RAW_CSV`、`CLASS_FILE`、`STOPWORDS_FILE`、`FIG_DIR`、`data_file(name)`。

**模型侧**：`MODELS_DIR`（即 `models/`）与 `model_path("rf/model.pkl")` —— 可含子目录，越界会被拒绝；**目录不会自动创建，需要时自行 mkdir**。

**标签与契约**：`CLASS_ORDER`、`CLASS_TO_ID`、`NUM_CLASSES`、`SENTIMENT_MAP`、`COLUMNS`、`CAT_COL`、`LABEL_COL`、`SPLITS`、`CSV_ENCODING`、`TXT_ENCODING`。

**超参建议初值**（只是起点，按需覆盖）：`EPOCHS=3`、`BATCH_SIZE=32`、`LR=5e-5`、`MAX_LEN=256`、`RANDOM_SEED=42`。`MAX_LEN` 取 256 的理由见「已知局限」第 3 条。

> **为什么数据组的脚本不 import config**：`data_preprocess.py` 等脚本必须能**单独复制出去运行**；而 config 是模型组会持续修改的文件，让已冻结的数据流水线依赖它，等于把外部改动风险引到已经冻结的资产上。所以那边保持自包含 —— 两边定位路径的方式不同，但指向同一份数据。

### 模型训练

模型训练脚本由模型组提供（随机森林 / FastText / BERT），尚未加入本仓库。数据准备完成后可直接使用 `final_data/` 下的文件：

- **FastText / 传统模型**：用 `train.txt` / `val.txt` / `test.txt`（每行 `文本<Tab>类别编号`）
- **BERT 微调**：用 `train.csv` / `val.csv` / `test.csv`（列 `review,label,cat_l1`）

## ⚠️ 已知局限与注意事项

以下几点是数据本身的属性，写报告和答辩时建议主动说明，避免被追问。数字均基于当前交付数据（62,600 条）实测：

1. **「乳制品」子类可被单个词识别。** 原始数据中该类的类名就是品牌名「蒙牛」，重命名为「乳制品」后，该子类仍有 **99.8%** 的评论字面包含「蒙牛」（2,033 条，现并入食品饮料大类）。因此涉及该子类的样本存在捷径特征，相关指标会偏高，不宜作为模型泛化能力的证据。

2. **关键词基线不可忽视。** 用各类别高频词做简单的关键词打分分类、**不做任何训练**，一级大类即可达到 **47.8%** 准确率；情感任务用「好评/满意/不错/垃圾/太差/失望」等词做计数规则可达 **78.15%**。报告中应把这两个基线一并列出，否则模型效果会显得虚高。

3. **建议 `max_len` 取 256 而非 128。** 有 **10.2%** 的评论超过 128 字，且这些正是信息量较大的长评论（图书文娱类中位数 145 字、本地生活类 69 字），用 128 会把它们整条截断。超过 510 字的仅 **0.49%**，BERT 的 512 上限不构成问题。

4. **存在跨类别的重复文本与少量 train/test 重叠。** 清洗后有 **27 条文本、58 行**（0.09%）出现在多个一级大类下，多为「一般一般一般」这类不含品类信息的套话；另有 **1 条**文本同时带两种情感标签。由于按（大类, 文本）去重、跨大类的同一句话会被保留，它在随机划分后会落到不同集合：**test 有 3 行、val 有 10 行与 train 存在完全相同的文本**。影响上限约 0.5 个百分点，不足以显著抬高准确率，但报告里不宜以个别模板句作为泛化能力的证据。当前选择保留这些行（它们是真实存在的评论），如需严格消除可再按文本做全局去重。

5. **`家用电器` 类的指标统计上很脆弱。** 该类只有 544 条，且 **81.6% 为正面**（全数据集唯一明显偏斜的类别）；按（大类, 情感）分层后，它的「负面」格子仅 train 70 / **val 15 / test 15** 条，n=15 时 95% 置信区间约 ±25 个百分点，**per-cell 指标基本是噪声**。同时 macro-F1 会被这个 0.87% 的小类主导。报告中给出 per-class 指标时必须同时列出 support（样本数），不要只报一个 macro-F1。

6. **仍残留少量未遮盖的姓名、地址与个别订单号。** 清洗遮盖了手机号、座机、服务热线、邮箱、QQ/微信、订单号等**号码类**信息（共 80 行，含「订单号是…」「…是我的订单编号」这类写法），但以下两类未处理：
   - **个人姓名与具体地址**：约 20 行含姓名、约 4 行含地址（例如配送员姓名、预约联系人、个别用户的收货地址）。未自动处理的原因是自动规则误报率过高 —— 例如「X先生」模式会命中 359 行，其中绝大多数是「南怀瑾先生」这类作者/公众人物，遮盖它们会破坏正常内容。
   - **个别订单号**：1 行中是同一句里用「和」并列的第二个订单号（`订单号[已脱敏]和59671095217`）。处理它需要针对连词写专门规则，收益不抵复杂度，故保留。

   如需彻底脱敏，应人工核对后定向处理。

> 以上所有数字均可用 `python src/data_audit.py` 一键复核（只读脚本，不修改任何文件）。

## 📝 许可证

本项目仅用于学习交流

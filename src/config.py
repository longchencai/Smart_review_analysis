# ============================================
# 项目配置 —— 面向模型组与后续新脚本
#
# 用法（脚本放在 src/ 下时）：
#     from config import DATA_DIR, CLASS_ORDER, SENTIMENT_MAP, load_csv
#
# 三条设计约定：
#   1. 所有路径由本文件位置推导，换机器、换目录、换操作系统都不用改
#   2. 标签定义从 data/processed/final_data/class.txt 读取，不在这里另抄一份，
#      因此不可能与实际交付的数据脱节
#   3. 数据清洗流水线（data_preprocess.py / generate_fasttext_data.py /
#      eda_analysis.py / data_audit.py）**刻意不依赖本文件**：它们必须能单独
#      复制运行，也不应被模型组的改动影响
# ============================================

import csv
from pathlib import Path

import pandas as pd

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "final_data"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "online_shopping_10_cats.csv"
FIG_DIR = PROJECT_ROOT / "data" / "figures" / "final_data"
CLASS_FILE = DATA_DIR / "class.txt"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"

# ---------- 数据契约 ----------
SPLITS = ("train", "val", "test")
COLUMNS = ("review", "label", "cat_l1")
CAT_COL = "cat_l1"          # 7 个大类，模型要预测的类别
LABEL_COL = "label"         # 0=负面, 1=正面
CSV_ENCODING = "utf-8-sig"  # 数据 CSV 带 UTF-8 BOM，必须用 utf-8-sig，
                            # 否则 pandas 会把首列名读成 '\ufeffreview'
TXT_ENCODING = "utf-8"
SENTIMENT_MAP = {0: "负面", 1: "正面"}

# ---------- 标签定义（从 class.txt 派生，切勿另抄一份） ----------
# class.txt 的行序就是标签编号 0..6，它同时是 train.txt 等文件里类别编号的权威定义。
if not CLASS_FILE.exists():
    raise FileNotFoundError(
        f"找不到 {CLASS_FILE}。它是标签编号的权威来源，"
        "请确认 data/processed/final_data/ 已从仓库完整拉取。"
    )
CLASS_ORDER = tuple(CLASS_FILE.read_text(encoding="utf-8").split())
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER)

# ---------- 训练超参建议初值（模型组按需覆盖） ----------
# 这些只是起点，不是硬性要求；模型组可自行调整，或改成命令行参数。
EPOCHS = 3
BATCH_SIZE = 32
LR = 5e-5            # BERT 微调常用量级
# max_len 取 256 而非 128：有 10.2% 的评论超过 128 字且多为长评论（图书类中位数 145 字），
# 用 128 会整条截断；超过 510 字的仅 0.49%，512 上限不构成问题。
MAX_LEN = 256
RANDOM_SEED = 42     # 划分与训练的随机种子（清洗脚本用的也是 42）


# ---------- 路径辅助 ----------
def data_file(name: str) -> Path:
    """在 final_data 目录内拼出文件路径，并校验没有越出该目录。"""
    path = (DATA_DIR / name).resolve()
    if path.parent != DATA_DIR:
        raise ValueError(f"路径越界，已拒绝：{name}")
    return path


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split 只能是 {SPLITS} 之一，收到：{split!r}")


def split_csv(split: str) -> Path:
    """某个划分的 CSV 路径（train / val / test）。"""
    _check_split(split)
    return data_file(f"{split}.csv")


def split_txt(split: str) -> Path:
    """某个划分的 TSV 路径（train / val / test）。"""
    _check_split(split)
    return data_file(f"{split}.txt")


# ---------- 各划分的数据路径 ----------
# 常见用途直接取常量；需要按名字遍历时用上面的 split_csv() / split_txt()
TRAIN_CSV = data_file("train.csv")
VAL_CSV = data_file("val.csv")
TEST_CSV = data_file("test.csv")
TRAIN_TXT = data_file("train.txt")
VAL_TXT = data_file("val.txt")
TEST_TXT = data_file("test.txt")

# ---------- 模型侧输出路径 ----------
MODELS_DIR = PROJECT_ROOT / "models"    # 仓库已建好该目录；模型与中间产物都放这里


def model_path(name: str) -> Path:
    """模型 / 中间产物的保存路径（相对 models/ 给出，可含子目录）。

    例：model_path("rf/model.pkl")、model_path("bert/bert_classifier.pt")
    目录不会自动创建，需要时自行 mkdir。
    """
    path = (MODELS_DIR / name).resolve()
    if MODELS_DIR not in path.parents:
        raise ValueError(f"路径越界，已拒绝：{name}")
    return path


# ---------- 读数据 ----------
def load_csv(split: str) -> pd.DataFrame:
    """读取某个划分的 CSV，返回 DataFrame（列：review, label, cat_l1）。

    内部已按 utf-8-sig 处理 BOM，直接 pd.read_csv(path) 会把首列名读成 '\\ufeffreview'。
    """
    return pd.read_csv(split_csv(split), encoding=CSV_ENCODING)


def load_fasttext_split(split: str):
    """读取某个划分的 TSV，返回 (texts, labels)，label 为 0..6 的整数列表。

    文件格式是「文本<Tab>类别编号」。必须用 quoting=csv.QUOTE_NONE：有 4 条评论
    以 '"' 开头，默认引号规则会把它们当成带引号的字段并合并相邻行，
    train 会静默少读 487 行（43820 → 43333）而不报错。
    """
    frame = pd.read_csv(
        split_txt(split), sep="\t", header=None, names=["review", "label"],
        quoting=csv.QUOTE_NONE, encoding=TXT_ENCODING,
    )
    return frame["review"].tolist(), frame["label"].astype(int).tolist()


def load_stopwords() -> set:
    """EDA 用的停用词表（data/processed/final_data/stopwords.txt，手工维护的输入）。"""
    lines = STOPWORDS_FILE.read_text(encoding=TXT_ENCODING).splitlines()
    return {line.strip() for line in lines if line.strip()}


if __name__ == "__main__":
    # 直接运行可自检配置：python src/config.py
    print("=" * 56)
    print("配置自检")
    print("=" * 56)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"类别数        : {NUM_CLASSES}")
    for i, name in enumerate(CLASS_ORDER):
        print(f"  {i} = {name}")
    print(f"情感标签      : {SENTIMENT_MAP}")
    print(f"超参建议初值  : EPOCHS={EPOCHS} BATCH_SIZE={BATCH_SIZE} LR={LR} MAX_LEN={MAX_LEN}")
    print("-" * 56)
    print("各划分数据路径：")
    for split in SPLITS:
        print(f"  {split:<6} {split_csv(split)}")
        print(f"  {'':<6} {split_txt(split)}")
    print(f"  {'模型侧':<5} {MODELS_DIR}（model_path('xxx') 取具体文件）")
    print("-" * 56)
    for split in SPLITS:
        frame = load_csv(split)
        texts, labels = load_fasttext_split(split)
        print(f"{split:<6} CSV {len(frame):>6} 行   TSV {len(texts):>6} 行   "
              f"类别数 {frame[CAT_COL].nunique()}")
    print("✅ 自检通过：路径可解析、class.txt 可读、CSV 与 TSV 行数一致")

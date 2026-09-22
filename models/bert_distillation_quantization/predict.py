# ============================================================
# 预测入口：封装成后端 API 可直接调用的推理器
#
# 同时支持两种学生权重：
#   · fp32（model/student_bert_4l384.pt）      —— 可跑 GPU / CPU
#   · int8（model/student_bert_4l384_int8.pt） —— 只能跑 CPU（PyTorch int8 算子无 CUDA kernel）
#
# 用法：
#   from predict import get_predictor
#   predictor = get_predictor(quantized=True)      # 进程内单例，适合 FastAPI 启动时预热
#   predictor.predict("这个键盘手感很好，就是有点贵")
#   predictor.predict_batch(["好评", "很差"])
#
# 或命令行自测：
#   python predict.py                      # 用内置样例跑一遍（int8）
#   python predict.py --fp32               # 用 fp32 跑
#   python predict.py --text "自定义评论"
#
# 关于线程安全：模型在 eval 模式下、且全程 torch.no_grad()，前向不修改任何状态；
# HF tokenizer 是 Rust 实现，也是线程安全的。因此本类可以被 FastAPI 的多线程
# 工作池共享，不需要额外加锁。若换成自定义的、会改状态的预处理才需要加锁。
# ============================================================

import argparse
import os
import time

import torch
import torch.nn.functional as F

from bert_config import Config
from student_model import StudentBertMultiTask

config = Config()

SEP = "=" * 72


def _readable(text):
    """把评论压成一行，便于日志与 CLI 展示。"""
    return " ".join(str(text).split())


# 学生架构的参数量只跟结构有关，与权重无关，算一次缓存即可
_ARCH_PARAM_COUNT = None


def architectural_param_count():
    """返回学生架构的参数量。

    注意不要用 sum(model.parameters()) 去数量化模型：动态量化会把
    nn.Linear / nn.Embedding 替换成量化模块，它们不再是 nn.Parameter，
    实测只数得到 6912（真实是 15,560,457），会严重误导。
    量化不改变参数量，所以这里按 fp32 架构统计。
    """
    global _ARCH_PARAM_COUNT
    if _ARCH_PARAM_COUNT is None:
        probe = StudentBertMultiTask()
        _ARCH_PARAM_COUNT = sum(p.numel() for p in probe.parameters())
        del probe
    return _ARCH_PARAM_COUNT


class StudentPredictor:
    """蒸馏学生（含量化版）的推理封装。

    参数
    ----
    quantized : bool
        True 用 int8 量化权重（强制 CPU）；False 用 fp32 权重。
    device : str | torch.device | None
        fp32 时的推理设备；None 表示跟随 config.device。
        量化模型会忽略该参数并强制 CPU（并给出一条提示）。
    model_path : str | None
        自定义权重路径；None 表示按 quantized 选择 config 里的默认路径。
    top_k : int
        默认返回的候选类别个数（0 表示不返回）。可在单次调用里覆盖。
    """

    def __init__(self, quantized=True, device=None, model_path=None, top_k=0, batch_size=64):
        self.quantized = bool(quantized)
        self.top_k = int(top_k)
        # 内部分块大小：一次前向最多处理这么多条。
        # 不加分块的话，服务端收到上千条会把 (N, 256) 的输入张量一次性压进显存/内存直接 OOM。
        self.batch_size = max(1, int(batch_size))
        self.model_path = model_path or (
            config.quantized_model_path if self.quantized else config.student_model_path)

        if self.quantized:
            # 量化模型只能在 CPU 上跑：PyTorch 的 int8 量化算子没有 CUDA 实现。
            # 注意：对量化模型调用 .to('cuda') 不会立刻报错，错误会延迟到前向传播。
            if device is not None and torch.device(device).type != 'cpu':
                print(f"[predict] 量化模型仅支持 CPU，已忽略 device={device}")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device) if device is not None else config.device

        self.model = None
        self.load_seconds = None

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load(self):
        """构建模型并载入权重。重复调用是幂等的。"""
        if self.model is not None:
            return self

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型权重：{self.model_path}")

        t0 = time.time()
        model = StudentBertMultiTask()

        if self.quantized:
            # 顺序很重要，写反会报 KeyError: '..._packed_params.dtype'
            #   加载 fp32 权重 = 建 fp32 模型 -> load_state_dict -> 就地 quantize_dynamic
            #   加载量化权重 = 先 quantize_dynamic 造同构空壳 -> 再 load_state_dict
            from quantize import quantize_model
            model = quantize_model(model)
            state = torch.load(self.model_path, map_location='cpu', weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"量化权重与模型结构不匹配：missing={list(missing)} unexpected={list(unexpected)}")
            model.to('cpu')
        else:
            state = torch.load(self.model_path, map_location='cpu', weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"fp32 权重与模型结构不匹配：missing={list(missing)} unexpected={list(unexpected)}")
            model.to(self.device)

        model.eval()
        self.model = model
        self.load_seconds = time.time() - t0
        return self

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    @staticmethod
    def _check_texts(texts):
        if not texts:
            raise ValueError("texts 不能为空")
        bad = [i for i, t in enumerate(texts) if not str(t).strip()]
        if bad:
            raise ValueError(f"第 {bad} 条文本为空或只有空白字符，请过滤后再调用")

    def _forward(self, texts):
        """返回 (cat_probs, sent_probs)，均为 CPU float32 numpy-free 张量。"""
        # 预处理必须与评估脚本 (dataloader_utils.my_collate_fn) 完全一致，
        # 否则离线指标与线上预测会对不上：
        #   max_length = config.distill_max_len, padding='max_length', truncation=True
        enc = config.bert_tokenizer(
            list(texts),
            max_length=config.distill_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            cat_logits, sent_logits = self.model(enc)
        return (F.softmax(cat_logits.float(), dim=-1).cpu(),
                F.softmax(sent_logits.float(), dim=-1).cpu())

    def _forward_chunked(self, texts, batch_size):
        """分块前向，避免一次性构造 (N, 256) 输入。"""
        cat_chunks, sent_chunks = [], []
        for i in range(0, len(texts), batch_size):
            c, s = self._forward(texts[i:i + batch_size])
            cat_chunks.append(c)
            sent_chunks.append(s)
        cat = cat_chunks[0] if len(cat_chunks) == 1 else torch.cat(cat_chunks, dim=0)
        sent = sent_chunks[0] if len(sent_chunks) == 1 else torch.cat(sent_chunks, dim=0)
        return cat, sent

    def predict(self, text, top_k=None, with_truncation=False):
        """单条评论预测，返回可直接 JSON 序列化的 dict。"""
        return self.predict_batch([text], top_k=top_k,
                                  with_truncation=with_truncation)[0]

    def predict_batch(self, texts, top_k=None, with_truncation=False, batch_size=None):
        """批量预测。批量调用比逐条调用快得多；内部按 batch_size 自动分块。"""
        self.load()
        texts = [str(t) for t in texts]
        self._check_texts(texts)

        k = self.top_k if top_k is None else int(top_k)
        k = max(0, min(k, config.cat_class_num))
        bs = max(1, int(batch_size or self.batch_size))

        # 截断标记要额外做一次不截断的分词，故默认关闭（会对小模型延迟产生可见影响）
        lengths = None
        if with_truncation:
            lengths = [len(config.bert_tokenizer.encode(t, truncation=False)) for t in texts]

        cat_probs, sent_probs = self._forward_chunked(texts, bs)

        out = []
        for i, raw in enumerate(texts):
            cp = cat_probs[i]
            sp = sent_probs[i]
            cat_id = int(cp.argmax())
            sent_id = int(sp.argmax())

            item = {
                "text": raw,
                "category": config.id2cat[cat_id],
                "category_id": cat_id,
                "category_confidence": round(float(cp[cat_id]), 6),
                "sentiment": config.id2sent[sent_id],
                "sentiment_id": sent_id,
                "sentiment_confidence": round(float(sp[sent_id]), 6),
            }
            if k:
                top = torch.topk(cp, k)
                item["top_categories"] = [
                    {"category": config.id2cat[int(j)], "probability": round(float(p), 6)}
                    for p, j in zip(top.values, top.indices)
                ]
            if lengths is not None:
                item["token_length"] = int(lengths[i])
                item["truncated"] = bool(lengths[i] > config.distill_max_len)
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # 元信息（给 /health、/info 用）
    # ------------------------------------------------------------------
    def info(self):
        self.load()
        return {
            "model_path": self.model_path,
            "quantized": self.quantized,
            "dtype": "int8" if self.quantized else "fp32",
            "device": str(self.device),
            "param_count": architectural_param_count(),
            "model_file_mb": round(os.path.getsize(self.model_path) / 1024 ** 2, 2),
            "student_layers": config.student_layers,
            "student_hidden": config.student_hidden,
            "max_len": config.distill_max_len,
            "batch_size": self.batch_size,
            "num_categories": config.cat_class_num,
            "categories": [config.id2cat[i] for i in range(config.cat_class_num)],
            "sentiments": [config.id2sent[i] for i in range(config.sent_class_num)],
            "load_seconds": round(self.load_seconds or 0.0, 3),
        }


# ---------------------------------------------------------------- 单例
_PREDICTORS = {}


def get_predictor(quantized=True, device=None, model_path=None, top_k=0, batch_size=64):
    """进程内单例。FastAPI 在启动时调一次即可，避免每个请求重复加载模型。"""
    key = (bool(quantized), str(device), model_path, int(top_k), int(batch_size))
    if key not in _PREDICTORS:
        _PREDICTORS[key] = StudentPredictor(
            quantized=quantized, device=device, model_path=model_path,
            top_k=top_k, batch_size=batch_size).load()
    return _PREDICTORS[key]


def predict(text, quantized=True, top_k=0, **kwargs):
    """便捷函数：单条预测。等价于 get_predictor(...).predict(text)。"""
    return get_predictor(quantized=quantized, top_k=top_k, **kwargs).predict(text, top_k=top_k)


# ---------------------------------------------------------------- CLI 自测
_SAMPLES = [
    "这个键盘手感很好，打字很舒服，就是价格有点贵，整体还算满意",
    "酒店位置偏僻，房间隔音差，前台态度也不好，不会再来了",
    "手机屏幕显示效果不错，但是电池续航太短了，充一次用不了半天",
    "书的内容很精彩，排版也舒服，物流很快，第二天就到了",
    "洗发水味道刺鼻，用完头皮发痒，果断退货",
    "热水器安装师傅很专业，出水温度稳定，洗澡很舒服",
]


def main():
    ap = argparse.ArgumentParser(description="学生模型预测入口自测")
    ap.add_argument('--fp32', action='store_true', help='用 fp32 权重（默认 int8）')
    ap.add_argument('--device', default=None, help='fp32 时的设备，如 cuda / cpu')
    ap.add_argument('--text', default=None, help='只预测这一条')
    ap.add_argument('--top-k', type=int, default=3, help='返回前 k 个候选类别（0=不返回）')
    args = ap.parse_args()

    quantized = not args.fp32
    predictor = get_predictor(quantized=quantized, device=args.device, top_k=args.top_k)
    meta = predictor.info()

    print(SEP)
    print(f"预测器自测  |  {'int8 量化' if quantized else 'fp32'}  |  设备 {meta['device']}")
    print(SEP)
    for key in ('model_path', 'model_file_mb', 'param_count', 'load_seconds'):
        print(f"  {key:16s}: {meta[key]}")
    print("-" * 72)

    texts = [args.text] if args.text else _SAMPLES
    t0 = time.time()
    results = predictor.predict_batch(texts, top_k=args.top_k)
    elapsed = time.time() - t0

    for r in results:
        print(f"  评论      : {_readable(r['text'])[:52]}")
        print(f"  商品大类  : {r['category']:<8} 置信度 {r['category_confidence']:.4f}")
        if r.get('top_categories'):
            cand = "  ".join(f"{c['category']}({c['probability']:.3f})" for c in r['top_categories'])
            print(f"  候选大类  : {cand}")
        print(f"  情感      : {r['sentiment']:<8} 置信度 {r['sentiment_confidence']:.4f}")
        print("-" * 72)

    print(f"  {len(texts)} 条共耗时 {elapsed * 1000:.0f} ms（{elapsed / len(texts) * 1000:.1f} ms/条，批量）")
    print(SEP)


if __name__ == '__main__':
    main()

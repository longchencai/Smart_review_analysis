# -*- coding: utf-8 -*-
"""学生模型在 test 集上的评估（本模块从未做过这件事）。
与项目评估口径一致：固定 padding 到 256，batch=64，macro-F1。
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score
from torch.ao.quantization import (default_dynamic_qconfig, float_qparams_weight_only_qconfig,
                                   quantize_dynamic)

warnings.filterwarnings("ignore")
torch.backends.quantized.engine = "onednn"

# 路径从脚本自身位置推算：本文件在 <项目根>/docs/model_audit/ 下，向上三级即项目根。
# 这样换机器、换 clone 目录都能直接跑，不需要改任何路径。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOD = os.path.join(ROOT, "models", "bert_distillation_quantization")
DATA = os.path.join(ROOT, "data", "processed", "final_data")


class StudentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = transformers.BertConfig.from_pretrained(MOD + r"\bert-base-chinese")
        cfg = transformers.BertConfig(
            vocab_size=base.vocab_size, hidden_size=384, num_hidden_layers=4,
            num_attention_heads=6, intermediate_size=1536, hidden_act=base.hidden_act,
            hidden_dropout_prob=base.hidden_dropout_prob,
            attention_probs_dropout_prob=base.attention_probs_dropout_prob,
            max_position_embeddings=base.max_position_embeddings,
            type_vocab_size=base.type_vocab_size, initializer_range=base.initializer_range,
            layer_norm_eps=base.layer_norm_eps, pad_token_id=base.pad_token_id)
        self.bert = transformers.BertModel(cfg)
        self.cat_linear = torch.nn.Linear(384, 7)
        self.sent_linear = torch.nn.Linear(384, 2)

    def forward(self, x):
        p = self.bert(**x)["pooler_output"]
        return self.cat_linear(p), self.sent_linear(p)


m32 = StudentModel()
m32.load_state_dict(torch.load(MOD + r"\model\student_bert_4l384.pt", map_location="cpu",
                               weights_only=True), strict=True)
m32.eval()
m8 = quantize_dynamic(StudentModel(), qconfig_spec={
    torch.nn.Linear: default_dynamic_qconfig,
    torch.nn.Embedding: float_qparams_weight_only_qconfig}, dtype=torch.qint8)
m8.load_state_dict(torch.load(MOD + r"\model\student_bert_4l384_int8.pt", map_location="cpu",
                              weights_only=True), strict=True)
m8.eval()

tok = transformers.BertTokenizer.from_pretrained(MOD + r"\bert-base-chinese")
class_txt = open(DATA + r"\class.txt", encoding="utf-8").read().splitlines()
cat2id = {c: i for i, c in enumerate(class_txt)}

for split in ("val", "test"):
    df = pd.read_csv(f"{DATA}\\{split}.csv", encoding="utf-8-sig")
    texts = [str(t) for t in df["review"]]
    y_cat = np.array([cat2id[str(x)] for x in df["cat_l1"]])
    y_sent = df["label"].to_numpy()
    print("=" * 72)
    print(f"【{split}】{len(texts)} 条")
    print("=" * 72)
    for tag, model in (("fp32", m32), ("int8", m8)):
        pc, ps = [], []
        with torch.no_grad():
            for s in range(0, len(texts), 64):
                enc = tok(texts[s:s + 64], max_length=256, padding="max_length",
                          truncation=True, return_tensors="pt")
                c, se = model(enc)
                pc.extend(torch.argmax(c, -1).tolist())
                ps.extend(torch.argmax(se, -1).tolist())
        cf = f1_score(y_cat, pc, average="macro", zero_division=0)
        sf = f1_score(y_sent, ps, average="macro", zero_division=0)
        print(f"  {tag:<5} 大类F1={cf:.4f}  情感F1={sf:.4f}  综合F1={(cf+sf)/2:.4f}")
    print()

# ============================================================
# 学生模型：SmallBERT 4L/384 双任务分类器
#
# 结构：transformers.BertModel(4 层 / 384 隐层 / 6 头 / FFN 1536)
#       + cat_linear(384 -> 7) + sent_linear(384 -> 2)
#
# 与教师的关系：
#   教师 BertModel 是 12 层 / 768 隐层 / 12 头 / FFN 3072
#   学生层数是教师的 1/3、宽度是教师的 1/2，注意力头维度同为 64（384/6 = 768/12）
#
# 关于初始化：本实现采用「随机初始化 + 输出层蒸馏」，
#   即学生只通过软标签/硬标签学习，不从教师搬运任何权重。
#   原因：宽度缩放后教师的权重矩阵形状与学生不匹配（768x768 vs 384x384），
#   需要额外的投影矩阵或按头切片，属于 TinyBERT 那一类的进阶做法；
#   本模块先把最小可用链路跑通，进阶初始化可作为后续消融实验。
# ============================================================

import torch
import transformers

from bert_config import Config

config = Config()


def build_student_bert_config(base_config=None):
    """按 config 里的学生规格，从教师骨架派生一个更小的 BertConfig。

    除了层数/宽度/头数/FFN，其余超参（词表、位置编码、激活函数、
    layer_norm_eps、初始化方差等）全部沿用教师，保证数值行为一致。
    """
    if base_config is None:
        base_config = transformers.BertConfig.from_pretrained(config.bert_base_chinese_path)
    return transformers.BertConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=config.student_hidden,
        num_hidden_layers=config.student_layers,
        num_attention_heads=config.student_heads,
        intermediate_size=config.student_ffn,
        hidden_act=base_config.hidden_act,
        hidden_dropout_prob=base_config.hidden_dropout_prob,
        attention_probs_dropout_prob=base_config.attention_probs_dropout_prob,
        max_position_embeddings=base_config.max_position_embeddings,
        type_vocab_size=base_config.type_vocab_size,
        initializer_range=base_config.initializer_range,
        layer_norm_eps=base_config.layer_norm_eps,
        pad_token_id=base_config.pad_token_id,
    )


class StudentBertMultiTask(torch.nn.Module):
    """学生：一个小 BERT 编码器 + 两个独立分类头（商品大类 / 情感）。

    前向签名与教师 MyBertMultiTaskClassifier 完全一致：
        输入 tokenizer 输出的 dict，返回 (cat_logits, sent_logits)
    这样蒸馏训练和评估代码可以无缝共用一个 dataloader 与 model_eval()。
    """

    def __init__(self):
        super().__init__()
        self.bert_config = build_student_bert_config()
        # 随机初始化的学生编码器（BertModel.__init__ 会做 post_init 权重初始化）
        self.bert = transformers.BertModel(self.bert_config)
        # 隐藏层 384 维 -> 两个任务
        self.cat_linear = torch.nn.Linear(config.student_hidden, config.cat_class_num)
        self.sent_linear = torch.nn.Linear(config.student_hidden, config.sent_class_num)

    def forward(self, x):
        # x: {'input_ids', 'token_type_ids', 'attention_mask'}
        result = self.bert(**x)
        pooled = result['pooler_output']                      # (batch, 384)
        cat_logits = self.cat_linear(pooled)                  # (batch, 7)
        sent_logits = self.sent_linear(pooled)                # (batch, 2)
        return cat_logits, sent_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def describe_student():
    """打印学生结构摘要，方便确认规格没写错。"""
    model = StudentBertMultiTask()
    total = count_parameters(model)
    emb = model.bert.embeddings.word_embeddings.weight.numel()
    print(f"学生结构: {config.student_layers} 层 / 隐层 {config.student_hidden} / "
          f"{config.student_heads} 头 / FFN {config.student_ffn}")
    print(f"参数量  : {total:,}  ({total / 1024 ** 2:.2f} M)")
    print(f"  其中词嵌入: {emb:,} ({emb / total * 100:.1f}%)")
    print(f"  分类头    : {model.cat_linear.weight.numel() + model.sent_linear.weight.numel():,}")
    return model


if __name__ == '__main__':
    m = describe_student()
    print(m)

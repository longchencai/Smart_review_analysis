# 导包
import transformers
import torch
import os


# 定义文件
# 项目根目录：从本文件位置自动推算（本文件位于 models/bert_distillation_quantization/bert_config.py）
# 向上两级即为项目根，避免硬编码某台机器的绝对路径，换机器 clone 后也能正确解析
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 本模块目录：models/bert_distillation_quantization/，模块自身资源都放在它下面
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

class Config():
    def __init__(self):
        print('正在初始化配置文件....')
        # 各种路径
        # TODO 1.原始数据路径
        # 基于脚本位置动态推算项目根（相对路径，不再写死绝对路径）
        self.root_path = _PROJECT_ROOT + '/'
        self.data_dir = self.root_path + 'data/processed/final_data/'
        self.train_path = self.data_dir + 'train.csv'
        self.val_path = self.data_dir + 'val.csv'
        self.test_path = self.data_dir + 'test.csv'
        # 停用词和类别路径
        self.stop_words_path = self.data_dir + 'stopwords.txt'
        self.class_path = self.data_dir + 'class.txt'

        # TODO 2.双任务相关配置
        # 商品大类分类任务：7 个一级大类
        self.cat_class_path = self.data_dir + 'class.txt'
        # 情感二分类任务：label 字段中 0=负面，1=正面
        self.sent_labels = ['负面评价', '正面评价']

        # TODO 3.BERT 相关路径和参数
        # 预训练权重就在本模块目录下的 bert-base-chinese/（已随仓库提交）
        self.bert_base_chinese_path = _MODEL_DIR + '/bert-base-chinese'
        # 兜底候选链：
        #   ① 本模块目录（正常情况）
        #   ② 项目根 models/bert-base-chinese（若手动拷贝过）
        #   ③ 都没有才从 HuggingFace 下载
        model_candidates = [
            self.bert_base_chinese_path,
            self.root_path + 'models/bert-base-chinese',
            'bert-base-chinese',
        ]
        model_path = None
        for cand in model_candidates:
            if os.path.exists(cand + '/config.json'):
                model_path = cand
                break
        if model_path is None:
            model_path = 'bert-base-chinese'
        print(f"使用 BERT 模型路径: {model_path}")
        # 提前加载 tokenizer 和 bert 模型对象
        # 加缓存：多个 Config() 实例（eval_utils / dataloader_utils / eval 脚本各自顶层都建了 Config）
        # 只会真正 from_pretrained 一次，避免重复加载把内存打爆导致 Rust tokenizer 报
        # "memory allocation of 2097152 bytes failed"
        if not hasattr(Config, '_model_cache'):
            Config._model_cache = {}
        if model_path not in Config._model_cache:
            print(f"首次加载 BERT 权重(已缓存复用): {model_path}")
            Config._model_cache[model_path] = (
                transformers.BertTokenizer.from_pretrained(model_path),
                transformers.BertModel.from_pretrained(model_path),
                transformers.BertConfig.from_pretrained(model_path),
            )
        self.bert_tokenizer, self.bert_model, self.bert_config = Config._model_cache[model_path]
        # 多任务 BERT 模型保存路径
        # 本模块训练出的权重，统一放在 models/bert_distillation_quantization/model/ 下
        self.bert_classifier_model_save_path = _MODEL_DIR + '/model/bert_multitask_classifier_model.pt'
        # 语义别名：上面这份权重就是「教师模型」，蒸馏时作为知识来源
        self.teacher_model_path = self.bert_classifier_model_save_path

        # ============================================================
        # TODO 4. 蒸馏（Distillation）配置 —— SmallBERT 4L/384 学生
        # ============================================================
        # 学生骨架：层数与宽度都小于教师；注意力头维度与教师同为 64（384/6 = 768/12）
        self.student_layers = 4
        self.student_hidden = 384
        self.student_heads = 6
        self.student_ffn = 1536

        # 蒸馏损失：loss = alpha * T^2 * KL(教师‖学生) + (1 - alpha) * CE(真实标签)
        #   T     : 温度。越大软标签越"平"、暗知识越明显，但噪声也越大
        #   alpha : 软标签权重。0 = 退化成普通监督训练；1 = 纯模仿教师（会继承教师的错误）
        # 注意 T^2 只乘在软标签项上：硬标签没经过温度缩放，不能乘
        self.distill_T = 2.0
        self.distill_alpha = 0.7
        self.distill_epochs = 4
        self.distill_batch_size = 32
        # 学生是随机初始化、从零训练的，必须用「预训练级」学习率。
        # 第一版误用了 BERT 微调的 5e-5，实测 3 个 epoch 后损失仍在快速下降、
        # 每轮指标都明显上涨，属于严重欠拟合，故上调到 3e-4。
        self.distill_lr = 3e-4
        self.distill_max_len = 256        # 必须与教师一致，否则软标签分布不可比
        self.distill_weight_decay = 0.01
        self.distill_grad_clip = 1.0

        # 教师软标签离线缓存：只存「原始 logits」，不预先除以 T，方便后续扫 T
        self.teacher_soft_label_path = _MODEL_DIR + '/cache/teacher_soft_labels.npz'
        self.cache_batch_size = 64        # 教师只做前向，可以开大

        # 蒸馏产出
        self.student_model_path = _MODEL_DIR + '/model/student_bert_4l384.pt'

        # ============================================================
        # TODO 5. 动态量化（Dynamic Quantization）配置
        # ============================================================
        # 量化后模型只能跑 CPU：PyTorch 的 int8 量化算子没有 CUDA 实现
        self.quantized_model_path = _MODEL_DIR + '/model/student_bert_4l384_int8.pt'
        # True 时只量化 nn.Linear —— 学生会丢掉一半以上压缩收益（其 Embedding 占 52%）
        self.quantize_only_linear = False
        # 本机 torch 构建仅提供 onednn 一个量化引擎（fbgemm / x86 不可用）
        self.quant_engine = 'onednn'

        # ============================================================
        # TODO 6. 评估与报告
        # ============================================================
        self.eval_batch_size = 64
        self.eval_result_path = _MODEL_DIR + '/eval_result.txt'
        self.csv_encoding = 'utf-8-sig'   # 数据 CSV 带 UTF-8 BOM，必须显式指定

        # 保证产物目录存在
        os.makedirs(_MODEL_DIR + '/model', exist_ok=True)
        os.makedirs(_MODEL_DIR + '/cache', exist_ok=True)

        # todo 参数
        # 前面适合 window   如果你是 mac 且支持 mps, 直接设置 'mps' 也行
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 5
        self.batch_size = 16
        self.lr = 5e-5
        # max_len 依据 data/processed/final_data/maxlen_stats.json 的真实 token 长度分布设定：
        # P50=36, P90=128, P95=182, P99=359。设 256 可覆盖 98.0% 的评论（仅截断 1.97%），
        # 显存/算力开销远小于 512（最长一句 2878，设成最大值会大量 padding 浪费）。
        self.max_len = 256
        # 商品大类类别数
        self.cat_class_num = 7
        # 情感类别数
        self.sent_class_num = 2

        # 类别名称映射（商品大类）
        self.id2cat = {index: line.strip() for index, line in enumerate(open(self.cat_class_path, 'r', encoding='utf8'))}
        self.cat2id = {v: k for k, v in self.id2cat.items()}
        # 情感名称映射
        self.id2sent = {0: '负面评价', 1: '正面评价'}
        self.sent2id = {'负面评价': 0, '正面评价': 1}

        print('配置文件初始化动作完成!')


if __name__ == '__main__':
    c = Config()
    print(c.bert_base_chinese_path)
    print(c.device)
    print(f"商品大类数: {c.cat_class_num}, 情感类别数: {c.sent_class_num}")
    print(f"id2cat: {c.id2cat}")
    print(f"id2sent: {c.id2sent}")

# 导包
import transformers
import torch
import os


# 定义文件
# 项目根目录：从本文件位置自动推算（本文件位于 models/bert/bert_config.py）
# 向上两级即为项目根，避免硬编码某台机器的绝对路径，换机器 clone 后也能正确解析
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config():
    # 教师骨架缓存（惰性加载，见下方 bert_model 属性）
    _teacher_model = None

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
        # 模型路径候选链（按优先级）：
        #   ① 当前项目 models/bert-base-chinese（若之后手动拷贝过）
        #   ② 原项目已下载好的路径（直接复用，免拷贝、省 C 盘空间）
        #   ③ 都没有才从 HuggingFace 下载
        self.bert_base_chinese_path = self.root_path + 'models/bert-base-chinese'
        # 可选兜底：若本机别处已经有下载好的 bert-base-chinese
        # （例如另一个项目里下载过，想跨项目复用、省一次下载），
        # 可以用环境变量 BERT_BASE_CHINESE_DIR 把目录指过来。
        # 原来这里直接写死了另一台机器上的绝对路径 —— 换机器虽然会被下面的
        # os.path.exists 跳过，但写死在代码里既误导别人又完全不可移植，改为环境变量注入。
        self.bert_base_chinese_path_legacy = os.environ.get('BERT_BASE_CHINESE_DIR', '')
        # 第三个候选：蒸馏模块目录下自带的 bert-base-chinese。
        # 它的 config.json / tokenizer.json / vocab.txt 都是**入库文件**（任何机器 clone 后都有），
        # 于是即使一个权重都没下载，教师模块也能拿到 tokenizer 和 config，
        # 「读数据 / 分词」这类不需要教师模型的用法可以直接跑起来。
        # 教师骨架本身（390 MB 权重）仍需单独获取，且只有真正访问 bert_model 时才会用到。
        #
        # ⚠️ 迁移提醒：这里只放**数据目录**，不 import 蒸馏模块的任何代码。
        #   如果你的机器上原本有一份带权重的 bert-base-chinese（旧版是写死在代码里的），
        #   现在需要显式用环境变量 BERT_BASE_CHINESE_DIR 指过去 —— 否则会优先命中
        #   下面这个 sibling 目录（它没有权重），教师模型反而加载不了。
        #   优先级：models/bert-base-chinese > $BERT_BASE_CHINESE_DIR > sibling > HuggingFace
        self.bert_base_chinese_path_sibling = (
            self.root_path + 'models/bert_distillation_quantization/bert-base-chinese')
        os.makedirs(self.root_path + 'models/', exist_ok=True)
        # 过滤掉空值：legacy 未用环境变量指定时是空串
        model_candidates = [c for c in (self.bert_base_chinese_path,
                                        self.bert_base_chinese_path_legacy,
                                        self.bert_base_chinese_path_sibling,
                                        'bert-base-chinese') if c]
        model_path = None
        for cand in model_candidates:
            if os.path.exists(cand + '/config.json'):
                model_path = cand
                break
        if model_path is None:
            model_path = 'bert-base-chinese'
        print(f"使用 BERT 模型路径: {model_path}")
        # 加载 tokenizer 和 BertConfig。
        # 这两个都很轻（几百 KB）且推理/数据处理路径本来就要用，直接在这里加载。
        # 加缓存：多个 Config() 实例（eval_utils / dataloader_utils / eval 脚本各自顶层都建了 Config）
        # 只会真正 from_pretrained 一次，避免重复加载把内存打爆导致 Rust tokenizer 报
        # "memory allocation of 2097152 bytes failed"
        #
        # 注意：教师骨架 BertModel（≈390 MB）**不在这里加载**，改为惰性属性 bert_model。
        # 原因：本模块的 Config() 是在每个脚本顶层就创建的，一旦在此处 eager 加载，
        #       「基座权重缺失」会让所有脚本（包括只用 tokenizer / 只用 dataloader 的）
        #       在 import 阶段就直接失败。改为惰性后，只有真正需要教师模型的地方才会加载。
        if not hasattr(Config, '_model_cache'):
            Config._model_cache = {}
        if model_path not in Config._model_cache:
            print(f"首次加载 BERT tokenizer/config(已缓存复用): {model_path}")
            Config._model_cache[model_path] = (
                transformers.BertTokenizer.from_pretrained(model_path),
                transformers.BertConfig.from_pretrained(model_path),
            )
        self.bert_tokenizer, self.bert_config = Config._model_cache[model_path]
        # 记下基座目录，供 bert_model 属性惰性加载时使用
        self._bert_base_path = model_path
        # 多任务 BERT 模型保存路径
        # 多任务 BERT 模型权重：与远程 models/fasttext/model/ 对齐，统一放在 models/bert/model/ 下
        self.bert_classifier_model_save_path = self.root_path + 'models/bert/model/bert_multitask_classifier_model.pt'

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

    # ============================================================
    # 教师骨架：惰性加载（用到才加载）
    #
    # 与本项目蒸馏模块（models/bert_distillation_quantization/bert_config.py）保持同一模式。
    #
    # 为什么不在 __init__ 里直接加载？
    #   本模块的 Config() 是在每个脚本顶层就创建的（dataloader_utils / bert_model_eval_utils /
    #   bert_classifier_model 等都是 `config = Config()`）。若在此 eager 加载 390 MB 的骨架，
    #   那么「基座权重缺失」会让这些脚本在 **import 阶段**就整体失败 ——
    #   哪怕只是想用 load_data_list() 读一下 csv、或只想拿 tokenizer，也会被一起打挂。
    #
    # 改成属性后：
    #   · 只读数据 / 只取 tokenizer 的用法完全不再依赖这 390 MB
    #   · 真正需要教师处（bert_classifier_model / bert_train / bert_eval_on_test /
    #     bert_predict_fun）在访问 config.bert_model 时才加载，并给出可读的报错
    # ============================================================
    @property
    def bert_model(self):
        if Config._teacher_model is None:
            path = getattr(self, '_bert_base_path', None)
            if not path:
                raise RuntimeError(
                    "Config 尚未初始化完成，无法定位基座权重路径；请先正常实例化 Config()")
            print(f"首次加载 BERT 教师骨架(惰性): {path}")
            try:
                Config._teacher_model = transformers.BertModel.from_pretrained(path)
            except OSError as e:
                # 分两种情形给不同的排查方向，避免把「联网失败」误报成「本地缺文件」。
                # （注意 requests 的 ProxyError / ConnectionError 也是 OSError 的子类，
                #   所以不能笼统地说成"目录下缺权重"。）
                if os.path.isdir(path):
                    reason = (
                        "基座权重文件缺失：该目录下没有 pytorch_model.bin / model.safetensors。\n"
                        f"  目录    : {path}\n"
                        "  获取方式: ① 运行 python docs/model_audit/fetch_base.py 自动下载\n"
                        "            ② 再用环境变量 BERT_BASE_CHINESE_DIR 指向下载目录，"
                        "或把权重文件直接放到上面的目录"
                    )
                else:
                    reason = (
                        "基座模型目录不存在，且未能从 HuggingFace 加载（可能是网络问题）。\n"
                        f"  期望目录: {path}\n"
                        "  建议    : 把 bert-base-chinese 放到该目录，"
                        "或用环境变量 BERT_BASE_CHINESE_DIR 指向已有目录"
                    )
                raise OSError(
                    "教师模型骨架加载失败。\n"
                    f"  {reason}\n"
                    "  提示    : 只读数据或只取 tokenizer 的用法不需要这个文件，"
                    "只有加载教师模型时才需要。\n"
                    f"  原始错误: {e}"
                ) from e
        return Config._teacher_model


if __name__ == '__main__':
    c = Config()
    print(c.bert_base_chinese_path)
    print(c.device)
    print(f"商品大类数: {c.cat_class_num}, 情感类别数: {c.sent_class_num}")
    print(f"id2cat: {c.id2cat}")
    print(f"id2sent: {c.id2sent}")

# 导包
import transformers
import torch
import os


# 定义文件
# 项目根目录：从本文件位置自动推算（本文件位于 models/bert/bert_config.py）
# 向上两级即为项目根，避免硬编码某台机器的绝对路径，换机器 clone 后也能正确解析
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        # 模型路径候选链（按优先级）：
        #   ① 当前项目 models/bert-base-chinese（若之后手动拷贝过）
        #   ② 原项目已下载好的路径（直接复用，免拷贝、省 C 盘空间）
        #   ③ 都没有才从 HuggingFace 下载
        self.bert_base_chinese_path = self.root_path + 'models/bert-base-chinese'
        # legacy：原项目已下载的权重（跨项目复用，本机绝对路径）
        # 其他机器若没有此目录会自动跳过，回退到从 HuggingFace 下载，故无需改成相对路径
        self.bert_base_chinese_path_legacy = 'D:/投满分1.0/_04_bert_base/bert-base-chinese'
        os.makedirs(self.root_path + 'models/', exist_ok=True)
        model_candidates = [self.bert_base_chinese_path, self.bert_base_chinese_path_legacy, 'bert-base-chinese']
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


if __name__ == '__main__':
    c = Config()
    print(c.bert_base_chinese_path)
    print(c.device)
    print(f"商品大类数: {c.cat_class_num}, 情感类别数: {c.sent_class_num}")
    print(f"id2cat: {c.id2cat}")
    print(f"id2sent: {c.id2sent}")

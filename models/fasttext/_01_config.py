import os

# TODO 0.重要：fasttext 底层是C++，在Windows下打不开含中文的路径。
#   本项目路径里带中文（投满分项目），所以这里统一把工作目录切到本模块目录，
#   之后凡是喂给 fasttext 的路径一律用"纯ASCII的相对路径"（见 TODO 3）。
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Config():
    def __init__(self):
        # TODO 1.项目根路径（本文件在 models/fasttext/ 下，往上退3层就是项目根目录）
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'

        # TODO 2.原始数据路径（数据组已经划分好的 train/val/test，csv格式）
        #   这些路径是给 pandas / open() 用的，含中文没问题
        self.train_path = self.root_path + 'data/processed/final_data/train.csv'
        self.dev_path = self.root_path + 'data/processed/final_data/val.csv'
        self.test_path = self.root_path + 'data/processed/final_data/test.csv'
        # 类别路径（7个一级大类）
        self.class_path = self.root_path + 'data/processed/final_data/class_names_l1.txt'

        # TODO 3.fasttext用的路径（相对本模块目录，纯ASCII，绝对不能写成绝对路径）
        #   处理后数据
        self.ft_train = 'processed_data/train_process_words.txt'
        self.ft_dev = 'processed_data/dev_process_words.txt'
        self.ft_test = 'processed_data/test_process_words.txt'
        # 模型保存路径（只保留一个模型）
        self.ft_model_default = 'model/ft_joint_default.bin'

        # 另外存一份绝对路径，方便在别处打印/调试用（不要传给fasttext）
        self.process_dir = self.root_path + 'models/fasttext/processed_data/'
        self.model_dir = self.root_path + 'models/fasttext/model/'

        # TODO 4.标签映射（方案B：单模型联合，一行写两个标签）
        #   商品大类用 __label__c{id} 表示，情感用 __label__s{id} 表示
        #   加前缀是为了预测时能把两个任务的标签拆开
        self.id2class = {index: line.strip() for index, line in enumerate(open(self.class_path, 'r', encoding='utf8'))}
        self.class2id = {name: index for index, name in self.id2class.items()}
        self.id2sentiment = {0: '负面', 1: '正面'}

        # 保存目录不存在就自动创建
        os.makedirs('processed_data', exist_ok=True)
        os.makedirs('model', exist_ok=True)


# 此处测试必须加main,不加main,其他位置导包的时候自动调用测试代码
if __name__ == '__main__':
    c = Config()
    print('工作目录:', os.getcwd())
    print('原始数据:', c.train_path)
    print('fasttext数据(相对):', c.ft_train)
    print('fasttext模型(相对):', c.ft_model_default)
    print('商品大类:', c.id2class)
    print('情感标签:', c.id2sentiment)

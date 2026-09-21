import os


class Config():
    def __init__(self):
        #项目根路径(本文件在 项目根/models/rf/config.py,向上三级就是项目根;用/拼接,和示例写法保持一致)
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace(os.sep, '/') + '/'

        #各种路径
        # 原始数据路径(AI8_Project1 的数据集是 csv,列名:review,label,cat_l1;label 就是情感二分类标签)
        self.train_path = self.root_path + 'data/processed/final_data/train.csv'
        self.test_path = self.root_path + 'data/processed/final_data/test.csv'
        self.dev_path = self.root_path + 'data/processed/final_data/val.csv'

        # 停用词和类别路径
        self.stop_words_path = self.root_path + 'data/processed/final_data/stopwords.txt'
        self.class_path = self.root_path + 'data/processed/final_data/class.txt'

        # 随机森林处理后数据存放路径(三列:words,cat_label,sent_label)
        self.process_train_path = self.root_path + 'models/rf/processed_data/train.txt'
        self.process_test_path = self.root_path + 'models/rf/processed_data/test.txt'
        self.process_dev_path = self.root_path + 'models/rf/processed_data/val.txt'

        # 随机森林模型存放路径(7分类)
        self.rf_save_model_path = self.root_path + 'models/rf/model/rf_model.pkl'
        self.tfidf_save_path = self.root_path + 'models/rf/model/tfidf_model.pkl'

        # 随机森林模型存放路径(情感二分类)
        self.rf_sentiment_save_model_path = self.root_path + 'models/rf/model/rf_model_sentiment.pkl'
        self.tfidf_sentiment_save_path = self.root_path + 'models/rf/model/tfidf_model_sentiment.pkl'

        # 分词产物里两个任务各自的标签列名
        self.cat_label_col = 'cat_label'
        self.sentiment_label_col = 'sent_label'

        # 情感二分类 索引 -> 类别名(原始数据 label 列:0=负面,1=正面)
        self.sentiment_id2class = {0: '负面', 1: '正面'}


# print(__name__) # 当前文件运行就是__main__,其他位置导包是模块名
# 此处测试必须加main,不加main,其他位置导包的时候自动调用测试代码
if __name__ == '__main__':
    c = Config()
    print(c.train_path)
    print(c.test_path)
    print(c.dev_path)
    print(c.process_train_path)
    print(c.process_test_path)
    print(c.process_dev_path)
    print(c.stop_words_path)
    print(c.class_path)
    print(c.rf_save_model_path)
    print(c.tfidf_save_path)
    print(c.rf_sentiment_save_model_path)
    print(c.tfidf_sentiment_save_path)
    print(c.cat_label_col, c.sentiment_label_col)
    print(c.sentiment_id2class)

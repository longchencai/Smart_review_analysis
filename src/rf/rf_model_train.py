# 导包
import os
import time
import pickle

import pandas as pd
import sklearn
# 示例里只写了 import sklearn 就直接访问 sklearn.xxx 子模块,部分 scikit-learn 版本下子模块不会被自动导入,
# 这里显式导入保证 sklearn.xxx 一定可用(其余写法与示例保持一致)
import sklearn.ensemble
import sklearn.feature_extraction.text
import sklearn.model_selection
from config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1.提前创建配置对象
config = Config()
# 2.读取分词后的数据文件(本项目 train.txt 里同时带7分类标签和情感二分类标签)
df_data = pd.read_csv(config.process_train_path, sep='\t')
# 3.单独获取特征(两个任务共用同一份文本特征)
x_words = df_data['words']
# 4.提前把停用词读取放到列表中
stop_words = [line.strip() for line in open(config.stop_words_path, 'r', encoding='utf8')]


# 定义训练函数:7分类和情感二分类各调用一次
def train_model(label_col, tfidf_save_path, rf_save_model_path):
    # 1.单独获取标签
    y_labels = df_data[label_col]
    # 2.使用TFIDF把文本特征转为数值特征
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stop_words)
    words_feature = tfidf.fit_transform(x_words)
    # 3.随机森林模型训练
    # 3.1 先切割数据
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        words_feature, y_labels, test_size=0.2, random_state=666)
    # 3.2 创建随机森林模型
    # n_estimators=100个决策树, verbose=2打印详细日志, n_jobs=-1使用本地电脑所有的处理器
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=-1)
    # 3.3 模型训练
    start_time = time.time()
    print(f'开始训练了...({label_col})')
    model.fit(X_train, y_train)
    print(f'训练结束,耗时{time.time() - start_time}')
    # 4.随机森林模型预测
    y_pred = model.predict(X_test)
    # 5.随机森林模型评估
    print(f"准确率:{accuracy_score(y_test, y_pred)}")
    print(f"精确率:{precision_score(y_test, y_pred, average='macro')}")
    print(f"召回率:{recall_score(y_test, y_pred, average='macro')}")
    print(f"f1分数:{f1_score(y_test, y_pred, average='macro')}")
    # 6.保存模型
    os.makedirs(os.path.dirname(rf_save_model_path), exist_ok=True)
    with open(tfidf_save_path, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(rf_save_model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'tfidf和rf模型保存成功!({label_col})')


if __name__ == '__main__':
    # 训练7分类模型
    train_model(config.cat_label_col, config.tfidf_save_path, config.rf_save_model_path)
    # 训练情感二分类模型
    train_model(config.sentiment_label_col, config.tfidf_sentiment_save_path, config.rf_sentiment_save_model_path)

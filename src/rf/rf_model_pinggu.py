# 导包
import pickle

import pandas as pd
from config import Config
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1.创建配置对象
config = Config()
# 2.读取测试集分词后数据(两个任务的标签在同一份文件的 cat_label / sent_label 两列)
df_data = pd.read_csv(config.process_test_path, sep='\t')


# 定义评估函数:7分类和情感二分类各调用一次
def pinggu_model(label_col, tfidf_save_path, rf_save_model_path, task_name):
    # 1.单独获取特征和标签
    x_words = df_data['words']
    y_label = df_data[label_col]
    # 2.加载tfidf然后把文本数据转为数值特征
    with open(tfidf_save_path, 'rb') as f:
        tfidf = pickle.load(f)
    new_x_words = tfidf.transform(x_words)
    # 3.加载rf模型然后预测
    with open(rf_save_model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(new_x_words)
    # 4.评估
    print(f'========== {task_name} ==========')
    print(f"准确率:{accuracy_score(y_label, y_pred)}")
    print(f"精确率:{precision_score(y_label, y_pred, average='macro')}")
    print(f"召回率:{recall_score(y_label, y_pred, average='macro')}")
    print(f"f1分数:{f1_score(y_label, y_pred, average='macro')}")


if __name__ == '__main__':
    # 评估7分类模型
    pinggu_model(config.cat_label_col, config.tfidf_save_path, config.rf_save_model_path, '7分类')
    # 评估情感二分类模型
    pinggu_model(config.sentiment_label_col, config.tfidf_sentiment_save_path, config.rf_sentiment_save_model_path, '情感二分类')

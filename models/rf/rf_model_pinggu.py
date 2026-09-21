# 导包
import pickle

import pandas as pd
from config import Config
import joblib
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

# 1.创建配置对象
config = Config()
# 2.读取测试集分词后数据(两个任务的标签在同一份文件的 cat_label / sent_label 两列)
df_data = pd.read_csv(config.process_test_path, sep='\t')

# 3.类别名:class.txt 的行号就是 cat_label 的编号;情感标签名从 config 里取
class_names = [line.strip() for line in open(config.class_path, 'r', encoding='utf8')]
sentiment_names = [config.sentiment_id2class[index] for index in sorted(config.sentiment_id2class)]

# 4.评估结果文本,两个任务都统计完后统一打印并写入 eval_result.txt
lines = []


# 定义评估函数:7分类和情感二分类各调用一次
def pinggu_model(label_col, tfidf_save_path, rf_save_model_path, task_name, target_names):
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
    # 预测阶段关掉训练日志、并行数设为1:测试集不大,单线程足够,也不会刷屏
    model.set_params(n_jobs=1, verbose=0)
    y_pred = model.predict(new_x_words)
    # 4.评估:macro 每个类别平等看待,weighted 按各类样本数加权
    acc = accuracy_score(y_label, y_pred)
    p_macro = precision_score(y_label, y_pred, average='macro', zero_division=0)
    r_macro = recall_score(y_label, y_pred, average='macro', zero_division=0)
    f_macro = f1_score(y_label, y_pred, average='macro', zero_division=0)
    p_weighted = precision_score(y_label, y_pred, average='weighted', zero_division=0)
    r_weighted = recall_score(y_label, y_pred, average='weighted', zero_division=0)
    f_weighted = f1_score(y_label, y_pred, average='weighted', zero_division=0)

    # 5.控制台打印(保持原有风格)
    print(f'========== {task_name} ==========')
    print(f"准确率:{acc}")
    print(f"精确率:{p_macro}")
    print(f"召回率:{r_macro}")
    print(f"f1分数:{f_macro}")

    # 6.追加到结果文本,格式和 fasttext 的 eval_result 保持一致
    lines.append('=' * 78)
    lines.append(f'【{task_name}】')
    lines.append('=' * 78)
    lines.append(f'  样本数      : {len(y_label)}')
    lines.append(f'  准确率      : {acc:.4f}')
    lines.append(f'  精确率 macro: {p_macro:.4f}    weighted: {p_weighted:.4f}')
    lines.append(f'  召回率 macro: {r_macro:.4f}    weighted: {r_weighted:.4f}')
    lines.append(f'  F1值   macro: {f_macro:.4f}    weighted: {f_weighted:.4f}')
    lines.append('')
    lines.append('  各类别明细：')
    lines.append(classification_report(y_label, y_pred, labels=list(range(len(target_names))),
                                       target_names=target_names, digits=4, zero_division=0))
    lines.append('')
    return acc, f_macro


if __name__ == '__main__':
    # 1.写表头:模型和测试集(路径相对于 models/rf)
    lines.append('评估模型: model/rf_model.pkl (商品大类 7分类)')
    lines.append('          model/rf_model_sentiment.pkl (情感倾向 二分类)')
    lines.append('测试集  : processed_data/test.txt')
    lines.append('')

    # 2.评估7分类模型
    acc_cat, f1_cat = pinggu_model(config.cat_label_col, config.tfidf_save_path,
                                   config.rf_save_model_path, '商品大类（7分类，主任务）', class_names)
    # 3.评估情感二分类模型
    acc_sent, f1_sent = pinggu_model(config.sentiment_label_col, config.tfidf_sentiment_save_path,
                                     config.rf_sentiment_save_model_path, '情感倾向（二分类，辅助任务）',
                                     sentiment_names)

    # 4.汇总
    lines.append('=' * 78)
    lines.append('【汇总】')
    lines.append('=' * 78)
    lines.append(f'  商品大类 : 准确率 {acc_cat:.4f}   宏F1 {f1_cat:.4f}')
    lines.append(f'  情感倾向 : 准确率 {acc_sent:.4f}   宏F1 {f1_sent:.4f}')
    lines.append('')
    lines.append('=' * 78)
    lines.append('【模型与特征信息】')
    lines.append('=' * 78)
    with open(config.tfidf_save_path, 'rb') as f:
        tfidf_cat = pickle.load(f)
    with open(config.process_train_path, 'r', encoding='utf8') as f:
        train_rows = sum(1 for _ in f) - 1  # 减掉表头
    lines.append('  模型     : RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)')
    lines.append('  特征     : TF-IDF 词表 %d 维(停用词 %d 个, ngram_range=(1, 1), norm=l2)'
                 % (len(tfidf_cat.vocabulary_), len(tfidf_cat.stop_words)))
    lines.append('  训练集   : processed_data/train.txt(%d 条)' % train_rows)

    # 5.打印并保存,方便写报告
    text = '\n'.join(lines)
    print(text)
    eval_result_path = config.root_path + 'models/rf/eval_result.txt'
    with open(eval_result_path, 'w', encoding='utf8') as fw:
        fw.write(text)
    print(f'\n结果已保存到: {eval_result_path}')

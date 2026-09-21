# 导包
import fasttext
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

from _01_config import Config

# TODO 提前创建配置对象
config = Config()

# TODO 1.加载模型
model = fasttext.load_model(config.ft_model_default)
print('模型加载完成:', config.ft_model_default)
print('===============================================================================')


# TODO 2.读取处理好的测试集
#   数据已经分好词、并且带了两个标签，形如：
#       __label__c3 __label__s1 键 盘 手 感 好
#   所以这里不需要再分词，直接读就行
def load_test_data(path):
    texts, y_cat, y_sent = [], [], []
    with open(path, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            # 只切两刀：前两个是标签，剩下全是文本
            label_c, label_s, text = line.split(maxsplit=2)
            y_cat.append(int(label_c.replace('__label__c', '')))
            y_sent.append(int(label_s.replace('__label__s', '')))
            texts.append(text)
    return texts, y_cat, y_sent


# TODO 3.批量预测，并按c/s前缀拆成两个任务
#   k=9 表示把全部9个标签和概率都取回来，再各挑概率最大的
def predict_all(texts):
    result = model.predict(texts, k=9)
    all_labels, all_probs = result[0], result[1]
    pred_cat, pred_sent = [], []
    for labels, probs in zip(all_labels, all_probs):
        best_c, best_s = -1, -1
        best_pc, best_ps = -1.0, -1.0
        for lb, p in zip(labels, probs):
            raw = lb.replace('__label__', '')       # 去掉 __label__ 前缀
            if raw.startswith('c') and p > best_pc:
                best_pc, best_c = float(p), int(raw[1:])
            elif raw.startswith('s') and p > best_ps:
                best_ps, best_s = float(p), int(raw[1:])
        pred_cat.append(best_c)
        pred_sent.append(best_s)
    return pred_cat, pred_sent


# TODO 4.算指标：准确率、精确率、召回率、F1
def show_metrics(y_true, y_pred, names, title, lines):
    acc = accuracy_score(y_true, y_pred)
    # macro：每个类别平等对待（数据不平衡时要看这个）
    p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # weighted：按每个类别的样本数加权（更接近整体感受）
    p_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    lines.append('=' * 78)
    lines.append(f'【{title}】')
    lines.append('=' * 78)
    lines.append(f'  样本数      : {len(y_true)}')
    lines.append(f'  准确率      : {acc:.4f}')
    lines.append(f'  精确率 macro: {p_macro:.4f}    weighted: {p_w:.4f}')
    lines.append(f'  召回率 macro: {r_macro:.4f}    weighted: {r_w:.4f}')
    lines.append(f'  F1值   macro: {f_macro:.4f}    weighted: {f_w:.4f}')
    lines.append('')
    lines.append('  各类别明细：')
    lines.append(classification_report(y_true, y_pred, labels=list(range(len(names))),
                                       target_names=names, digits=4, zero_division=0))
    lines.append('')
    return acc, f_macro


if __name__ == '__main__':
    # 读测试集
    texts, y_cat, y_sent = load_test_data(config.ft_test)
    print(f'测试集样本数: {len(texts)}')

    # 批量预测
    pred_cat, pred_sent = predict_all(texts)
    print('预测完成，开始统计指标...')
    print('===============================================================================')

    # 统计并打印
    lines = []
    lines.append(f'评估模型: {config.ft_model_default}')
    lines.append(f'测试集  : {config.ft_test}')
    lines.append('')

    acc_c, f1_c = show_metrics(y_cat, pred_cat, list(config.id2class.values()),
                               '商品大类（7分类，主任务）', lines)
    acc_s, f1_s = show_metrics(y_sent, pred_sent, list(config.id2sentiment.values()),
                               '情感倾向（二分类，辅助任务）', lines)

    lines.append('=' * 78)
    lines.append('【汇总】')
    lines.append('=' * 78)
    lines.append(f'  商品大类 : 准确率 {acc_c:.4f}   宏F1 {f1_c:.4f}')
    lines.append(f'  情感倾向 : 准确率 {acc_s:.4f}   宏F1 {f1_s:.4f}')

    text = '\n'.join(lines)
    print(text)

    # 保存到文件，方便写报告
    with open('eval_result.txt', 'w', encoding='utf8') as fw:
        fw.write(text)
    print('\n结果已保存到: eval_result.txt')

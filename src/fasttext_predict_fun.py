# 模型预测（双任务版本）
import fasttext
import jieba
import warnings
import os

warnings.filterwarnings('ignore')

# 1、加载两个模型（模块被导入时加载一次，服务启动后所有请求共用）
MODEL_DIR = "../models/fasttext"

# 自动查找最新的模型文件（按文件名排序取最后一个）
cat_models = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('fastText_cat_') and f.endswith('.bin')])
sent_models = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('fastText_sent_') and f.endswith('.bin')])

if not cat_models:
    raise FileNotFoundError(f"未找到商品大类 fastText 模型，请先在 {MODEL_DIR} 训练并保存模型")
if not sent_models:
    raise FileNotFoundError(f"未找到情感 fastText 模型，请先在 {MODEL_DIR} 训练并保存模型")

cat_model = fasttext.load_model(f"{MODEL_DIR}/{cat_models[-1]}")
sent_model = fasttext.load_model(f"{MODEL_DIR}/{sent_models[-1]}")
print(f"已加载商品大类模型: {cat_models[-1]}")
print(f"已加载情感模型: {sent_models[-1]}")


# 2、定义预测函数
def predict(data):
    # 与 generate_fasttext_data_multitask.py 的分词方式保持一致
    text = str(data["text"]).replace('\t', ' ').replace('\n', ' ')
    words = jieba.lcut(text)
    words = [w for w in words if w.strip()]
    words_str = ' '.join(words)

    # 商品大类预测
    cat_res = cat_model.predict(words_str)
    cat_pred_label = cat_res[0][0][9:]  # 去掉 __label__

    # 情感预测
    sent_res = sent_model.predict(words_str)
    sent_pred_label = sent_res[0][0][9:]  # 去掉 __label__

    # 封装结果并返回
    data["predict_category"] = cat_pred_label
    data["predict_sentiment"] = sent_pred_label
    return data


if __name__ == '__main__':
    # 单独运行本文件时才做一次测试预测；被 api.py 导入时不会执行
    print(predict({"text": "手机信号很好，拍照清晰，非常满意！"}))
    print(predict({"text": "快递太慢了，包装也破了，失望。"}))

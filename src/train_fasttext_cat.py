# 导入工具包
import fasttext
import datetime
import os

# 获取时间
 current_time = datetime.datetime.now().date().today().strftime("%Y%m%d")

# 模型保存路径
SAVE_DIR = "../models/fasttext"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1、模型训练（商品大类 7 分类）
model = fasttext.train_supervised(
    input=r'../data/processed/final_data/train_fasttext_cat.txt',
)

# 2、模型保存
save_path = f"{SAVE_DIR}/fastText_cat_{current_time}.bin"
model.save_model(save_path)
print(f"商品大类 fastText 模型已保存: {save_path}")

# 3、模型测试
print("\n商品大类模型测试评估开始...")
res = model.test(r'../data/processed/final_data/test_fasttext_cat.txt')
print(res)

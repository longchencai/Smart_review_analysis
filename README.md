# 电商评论智能分类系统

## 📖 项目简介

基于BERT的电商评论智能分类系统，通过分析用户评论内容，自动判断评论所属的商品类别，并预测情感倾向。

## 🎯 项目任务

### 主任务：商品类别9分类
根据评论内容，判断属于哪个商品类别：
- 书籍、平板、手机、水果、洗发水、蒙牛、衣服、计算机、酒店

### 辅助任务：情感正负二分类
判断评论是正面评价还是负面评价

## 🛠️ 技术栈

- **编程语言**：Python 3.9+
- **深度学习框架**：PyTorch
- **预训练模型**：BERT-base-chinese
- **传统机器学习**：Scikit-learn（随机森林）
- **快速文本分类**：FastText
- **Web部署**：FastAPI + HTML
- **版本管理**：Git + Gitee

## 📁 目录结构

```
ai8_-project1/
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/         # 清洗后的数据
│   └── figures/            # EDA可视化图表
├── src/                    # 代码目录
│   ├── data_preprocess.py  # 数据清洗脚本
│   └── eda_analysis.py    # EDA分析脚本
├── models/                # 训练好的模型文件
├── docs/                  # 项目文档
└── README.md              # 项目说明
```

## 👥 团队分工

| 小组 | 成员 | 职责 |
|---|---|---|
| 数据组 | 蔡隆宸、李欣祥 | 数据清洗、EDA分析、特征工程 |
| 模型组 | 待补充 | 随机森林、FastText、BERT模型训练与评估 |
| 优化部署组 | 待补充 | 模型蒸馏/量化、FastAPI部署、Web页面 |
| 文档组 | 待补充 | 项目文档、答辩PPT、答辩排练 |

## 📊 项目进度

- [x] 数据清洗与预处理
- [x] EDA探索性分析（9张可视化图表）
- [ ] 特征工程（分词 + TF-IDF）
- [ ] 基线模型训练（随机森林 + FastText）
- [ ] BERT模型训练与调优
- [ ] 模型对比与评估
- [ ] 模型优化（蒸馏/量化）
- [ ] FastAPI + Web部署
- [ ] 答辩PPT制作与排练

## 🚀 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### 数据预处理
```bash
cd src
python data_preprocess.py
```

### 模型训练
```bash
python train_bert.py
```

## 📝 许可证

本项目仅用于学习交流

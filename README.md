# 🏥 医疗数据分析与可视化平台

> 基于 Pima Indians Diabetes Dataset 的糖尿病数据分析与预测系统
> 医学信息工程专业 | 数据方向项目

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blueviolet)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-9cf)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

## 📖 项目简介

本项目是一个基于 **Pima Indians Diabetes Dataset**（皮马印第安人糖尿病数据集）的医疗数据分析与预测平台，使用 **Streamlit** 构建交互式Web应用。


## 🛠️ 技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 编程语言 | Python | 3.11.9 | 开发语言 |
| Web框架 | Streamlit | 1.31+ | 快速构建数据应用 |
| 数据处理 | Pandas | 2.0+ | 数据清洗与分析 |
| 数值计算 | NumPy | 1.24+ | 数值计算 |
| 可视化 | Matplotlib | 3.7+ | 基础绘图 |
| 可视化 | Seaborn | 0.12+ | 统计可视化 |
| 机器学习 | Scikit-learn | 1.3+ | 建模与评估 |
| 交互式图表 | Plotly | 5.18+ | 可选，交互式图表 |

## ✨ 功能亮点

### 📖 1. 数据集介绍
- 详细的数据来源说明
- 每个字段的医学含义解释
- 数据集基本统计信息
- 医学背景知识科普

### 📊 2. 数据概览
- 数据表格预览（可调行数）
- 各项指标统计描述（均值/标准差/分位数）
- 缺失值统计与可视化
- 患病比例分布分析

### 📈 3. 数据可视化
- **各指标分布直方图**：带 KDE 曲线，按患病分组对比
- **相关性热力图**：展示各特征之间的相关关系
- **患病 vs 未患病对比**：箱线图多维度对比
- **年龄分布与患病关系**：分年龄段患病率分析
- **特征箱线图**：离群值检测

### 🤖 4. 预测模型
- 逻辑回归算法建模
- 多维度评估指标展示：
  - 准确率 / 精确率 / 召回率 / F1分数
  - 混淆矩阵热力图
  - ROC曲线 + AUC值
- 特征重要性分析（系数解读）
- 详细的评估指标数学公式说明

### 🧮 5. 交互式风险计算器
- 实时输入8项指标
- 一键预测患病概率
- 风险等级分级提示（低/中/高）
- 个性化健康建议
- 各特征贡献度分析
- 3个典型案例快速测试按钮

## 🖼️ 效果截图


### 首页 - 数据集介绍
<!-- ![数据集介绍](./screenshots/intro.png) -->
<img width="2546" height="1320" alt="2f5429f2e34f721beb0896f298f1fe16" src="https://github.com/user-attachments/assets/946f575a-c020-4df7-b1a8-599248aa0f10" />


### 数据概览
<!-- ![数据概览](./screenshots/overview.png) -->
![数据概览占位图](https://via.placeholder.com/900x500?text=Data+Overview)

### 数据可视化
<!-- ![数据可视化](./screenshots/visualization.png) -->
<img width="2508" height="1312" alt="da1a70dbc74450061be28793a478bb53" src="https://github.com/user-attachments/assets/288c61ee-1ecb-438f-b753-18f48befdb3a" />
<img width="1513" height="1300" alt="1047d5216ba78383d0218ede24b31d5a" src="https://github.com/user-attachments/assets/4f82e160-fb50-40d5-b4c1-ad97d133369d" />



### 预测模型
<!-- ![预测模型](./screenshots/model.png) -->
![预测模型占位图](https://via.placeholder.com/900x500?text=Prediction+Model)

### 风险计算器
<!-- ![风险计算器](./screenshots/calculator.png) -->
<img width="2039" height="1280" alt="5361b23e6cef19dd5ebddda67be84a0a" src="https://github.com/user-attachments/assets/62e8e3bf-8f29-4044-9513-4f49b6d7e637" />

## 📚 数据集说明

### 数据集名称
**Pima Indians Diabetes Dataset**（皮马印第安人糖尿病数据集）

### 数据来源
- **原始来源**：美国国家糖尿病、消化和肾脏疾病研究所 (NIDDK)
- **经典平台**：UCI Machine Learning Repository / Kaggle / OpenML
- **数据大小**：768 条记录，8 个输入特征 + 1 个目标变量

### 字段说明

| 字段名 | 中文含义 | 单位 | 说明 |
|--------|---------|------|------|
| Pregnancies | 怀孕次数 | 次 | - |
| Glucose | 口服葡萄糖耐量试验2小时血糖 | mg/dL | 糖尿病最重要的指标 |
| BloodPressure | 舒张压 | mm Hg | - |
| SkinThickness | 三头肌皮褶厚度 | mm | 反映体脂情况 |
| Insulin | 2小时血清胰岛素 | mu U/ml | 存在大量缺失值 |
| BMI | 体重指数 | kg/m² | 体重/身高² |
| DiabetesPedigreeFunction | 糖尿病谱系函数 | - | 反映遗传风险 |
| Age | 年龄 | 岁 | - |
| Outcome | 是否患病 | 0/1 | **目标变量**: 1=患病, 0=未患病 |

### 数据特点
- 全部为女性（21岁以上 Pima 印第安族裔）
- 该族群糖尿病患病率较高，是研究糖尿病的经典数据集
- 部分特征存在 0 值（缺失值），需要预处理

## 🚀 运行步骤

### 环境要求
- Python 3.9 或更高版本
- pip 包管理工具

### 安装步骤

1. **克隆或下载项目**
   ```bash
   git clone <你的仓库地址>
   cd medical-data-analysis
   ```

2. **创建虚拟环境（推荐）**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

   如果安装速度慢，可以使用国内镜像源：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **运行项目**
   ```bash
   streamlit run app.py
   ```

5. **访问应用**
   - 启动后浏览器会自动打开
   - 默认地址：http://localhost:8501

### 注意事项

- 首次运行会自动从网络下载数据集（约 23KB），请确保网络通畅
- 如果无法下载数据集，程序会自动使用内置的模拟数据进行演示
- 运行中所有计算结果都会缓存，切换页面不会重复计算

## 🏗️ 项目结构

```
medical-data-analysis/
├── app.py                    # 主程序（单文件，包含所有页面）
├── requirements.txt          # 依赖包列表
├── README.md                 # 项目说明文档
└── screenshots/              # 截图文件夹（可自行创建）
    ├── intro.png
    ├── overview.png
    ├── visualization.png
    ├── model.png
    └── calculator.png
```

### 代码结构说明

`app.py` 采用模块化设计，主要包含以下函数：

| 函数 | 功能 |
|------|------|
| `load_data()` | 加载数据集（带缓存） |
| `train_model()` | 训练逻辑回归模型（带缓存） |
| `sidebar()` | 侧边栏导航 |
| `page_intro()` | 数据集介绍页面 |
| `page_overview()` | 数据概览页面 |
| `page_visualization()` | 数据可视化页面 |
| `page_model()` | 预测模型页面 |
| `page_calculator()` | 风险计算器页面 |
| `main()` | 主函数 |

## 💡 项目亮点

1. **🧬 医学专业结合**
   - 选用经典糖尿病数据集
   - 图表分析附带医学意义解读
   - 风险计算器有分级提示和健康建议
   - 评估指标解释包含临床意义（灵敏度/特异度）

2. **📊 完整的数据分析流程**
   - 数据探索 → 可视化 → 建模 → 部署
   - 涵盖数据分析全流程技能点
   - 多种图表类型展示可视化能力

3. **🤖 机器学习实战**
   - 逻辑回归模型，可解释性强
   - 完整的模型评估体系（准确率/精确率/召回率/F1/AUC/混淆矩阵）
   - 特征重要性分析
   - 数学公式说明体现理论功底

4. **🎯 交互式应用**
   - Streamlit 快速构建Web应用
   - 实时计算、即时反馈
   - 风险计算器非常有实用性
   - 侧边栏导航，多页面切换

5. **📝 代码规范**
   - 详细的中文注释
   - 模块化的函数设计
   - 缓存机制优化性能
   - 异常处理（数据下载失败有降级方案）

## 🔧 扩展思路

想让项目更丰富？可以考虑以下扩展方向：

- [ ] 增加决策树、随机森林、SVM等更多模型对比
- [ ] 增加特征工程（缺失值处理、特征组合等）
- [ ] 使用 Plotly 替代 Matplotlib 实现交互式图表
- [ ] 增加 K 折交叉验证
- [ ] 增加网格搜索调参
- [ ] 增加 SHAP 可解释性分析
- [ ] 部署到云端（Streamlit Community Cloud 免费部署）
- [ ] 增加更多数据集（如乳腺癌、心脏病等）
- [ ] 增加用户登录/数据保存功能

## 🤝 相关知识

### 什么是 Streamlit？
Streamlit 是一个开源的 Python 库，可以快速将数据脚本转化为可共享的 Web 应用。
它不需要前端知识，纯 Python 即可构建漂亮的交互式应用，是数据科学家展示成果的利器。

### 为什么用逻辑回归？
逻辑回归是医学统计中最常用的分类方法之一：
- **可解释性强**：每个特征的系数（OR值）有明确的医学意义
- **计算效率高**：适合小样本数据集
- **临床应用广**：很多临床预测模型（如风险评分）都是基于逻辑回归

## 📄 License

MIT License

---

<div align="center">
  <p>
    如果这个项目对你有帮助，欢迎点个 ⭐ Star 支持一下！
  </p>
  <p>
    <sub>lee</sub>
  </p>
</div>

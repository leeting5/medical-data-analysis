# -*- coding: utf-8 -*-
"""
🏥 医疗数据分析与可视化平台
基于 Pima Indians Diabetes Dataset 的糖尿病数据分析与预测

技术栈：Streamlit + Pandas + Matplotlib/Seaborn + Scikit-learn
作者：leeting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="糖尿病数据分析与预测平台",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体（用于matplotlib）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================
# 数据加载（缓存加速）
# ============================
@st.cache_data(ttl=3600)
def load_data():
    """
    加载 Pima Indians Diabetes 数据集
    优先从 sklearn/openml 获取，失败则使用备用方式
    """
    try:
        # 方式1：从 openml 获取
        dataset = fetch_openml(name='diabetes', version=1, as_frame=True, parser='liac-arff')
        df = dataset.frame
        # 统一列名
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        # 将 Outcome 转为数值类型
        df['Outcome'] = df['Outcome'].map({'tested_positive': 1, 'tested_negative': 0}).astype(int)
        return df
    except Exception as e:
        try:
            # 方式2：使用 UCI 数据集链接
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            df = pd.read_csv(url, names=columns)
            df['Outcome'] = df['Outcome'].astype(int)
            return df
        except Exception as e2:
            # 方式3：生成模拟数据（确保项目能运行）
            st.warning("无法从网络获取数据，使用内置模拟数据进行演示")
            np.random.seed(42)
            n_samples = 768
            df = pd.DataFrame({
                'Pregnancies': np.random.randint(0, 15, n_samples).astype(float),
                'Glucose': np.random.normal(120, 30, n_samples).astype(float),
                'BloodPressure': np.random.normal(70, 15, n_samples).astype(float),
                'SkinThickness': np.random.normal(29, 10, n_samples).astype(float),
                'Insulin': np.random.normal(80, 60, n_samples).astype(float),
                'BMI': np.random.normal(32, 7, n_samples).astype(float),
                'DiabetesPedigreeFunction': np.random.exponential(0.4, n_samples).astype(float),
                'Age': np.random.randint(21, 80, n_samples).astype(float),
                'Outcome': np.random.randint(0, 2, n_samples)
            })
            # 让数据更真实：葡萄糖高的更容易患病
            df.loc[df['Glucose'] > 140, 'Outcome'] = np.where(
                np.random.random((df['Glucose'] > 140).sum()) > 0.3, 1,
                df.loc[df['Glucose'] > 140, 'Outcome']
            )
            return df


@st.cache_resource
def train_model(df):
    """
    训练逻辑回归模型
    返回：模型、标准化器、评估指标
    """
    # 数据预处理
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练逻辑回归
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    # 计算 ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['roc_auc'] = auc(fpr, tpr)

    return model, scaler, metrics, X_train.columns.tolist()


# ============================
# 侧边栏导航
# ============================
def sidebar():
    with st.sidebar:
        st.title("🏥 糖尿病数据分析平台")
        st.markdown("---")

        page = st.radio(
            "导航菜单",
            ["📖 数据集介绍",
             "📊 数据概览",
             "📈 数据可视化",
             "🤖 预测模型",
             "🧮 风险计算器"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 关于项目")
        st.info(""" 
        技术栈：
        - Python 3.11.9
        - Streamlit
        - Pandas
        - Scikit-learn
        - Matplotlib / Seaborn
        """)

        st.markdown("---")
        st.caption("leeting")

    return page


# ============================
# 页面1：数据集介绍
# ============================
def page_intro(df):
    st.title("📖 数据集介绍")
    st.markdown("---")

    st.header("Pima Indians Diabetes Dataset")
    st.markdown("""
    这是一个经典的糖尿病预测数据集，最初由美国国家糖尿病、消化和肾脏疾病研究所收集。
    数据集包含了 **768 名** 至少 21 岁的 Pima 印第安女性患者的医学数据，
    目标是基于各项诊断指标预测患者是否患有糖尿病。

    """)

    st.subheader("📋 数据来源")
    st.info("""
    - **原始来源**：National Institute of Diabetes and Digestive and Kidney Diseases
    - **存储平台**：Kaggle / UCI Machine Learning Repository / OpenML
    - **数据集大小**：768 条记录，9 个特征
    """)

    st.subheader("🏷️ 字段说明")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        | 字段名 | 中文含义 | 单位 |
        |--------|---------|------|
        | Pregnancies | 怀孕次数 | 次 |
        | Glucose | 口服葡萄糖耐量试验中2小时的血糖浓度 | mg/dL |
        | BloodPressure | 舒张压 | mm Hg |
        | SkinThickness | 三头肌皮褶厚度 | mm |
        | Insulin | 2小时血清胰岛素 | mu U/ml |
        """)

    with col2:
        st.markdown("""
        | 字段名 | 中文含义 | 单位 |
        |--------|---------|------|
        | BMI | 体重指数 | kg/m² |
        | DiabetesPedigreeFunction | 糖尿病谱系函数（遗传风险） | - |
        | Age | 年龄 | 岁 |
        | **Outcome** | **是否患病（目标变量）** | **0=否, 1=是** |
        """)

    st.subheader("📊 数据集基本信息")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("样本总数", f"{len(df)}")
    col2.metric("特征数量", f"{df.shape[1]-1}")
    col3.metric("患病比例", f"{df['Outcome'].mean()*100:.1f}%")
    col4.metric("未患病比例", f"{(1-df['Outcome'].mean())*100:.1f}%")

    st.subheader("⚠️ 数据说明")
    st.warning("""
    注意：数据集中部分特征存在 0 值，在医学上属于不合理数据（如血压为0、血糖为0等），
    这些实际上代表缺失值。在实际项目中需要进行缺失值处理（删除/插值/标记）。
    本平台保留原始数据用于教学演示，同时提供缺失值统计供分析参考。
    """)


# ============================
# 页面2：数据概览
# ============================
def page_overview(df):
    st.title("📊 数据概览")
    st.markdown("---")

    # 数据预览
    st.subheader("📋 数据预览")
    n_rows = st.slider("显示行数", 5, 50, 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    # 统计描述
    st.subheader("📈 统计描述")
    desc = df.describe().T
    desc = desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    # 缺失值统计
    st.subheader("❓ 缺失值统计")
    st.markdown("将 0 值视为缺失值（除了 Pregnancies 和 Outcome）")

    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    missing_data = pd.DataFrame({
        '特征': zero_cols,
        '0值数量': [df[col].eq(0).sum() for col in zero_cols],
        '0值比例(%)': [round(df[col].eq(0).sum() / len(df) * 100, 2) for col in zero_cols]
    })
    st.dataframe(missing_data, use_container_width=True, hide_index=True)

    # 缺失值可视化
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_ratios = [df[col].eq(0).sum() / len(df) * 100 for col in zero_cols]
    bars = ax.bar(zero_cols, missing_ratios, color='#409eff', alpha=0.7)
    ax.set_xlabel('特征')
    ax.set_ylabel('缺失值比例 (%)')
    ax.set_title('各特征缺失值（0值）比例')
    ax.tick_params(axis='x', rotation=15)
    for bar, ratio in zip(bars, missing_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    # 患病分布
    st.subheader("👥 患病分布")
    col1, col2 = st.columns(2)

    with col1:
        outcome_counts = df['Outcome'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#67c23a', '#f56c6c']
        wedges, texts, autotexts = ax.pie(
            outcome_counts.values,
            labels=['未患病 (0)', '患病 (1)'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title('糖尿病患病比例分布')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Outcome', data=df, palette=['#67c23a', '#f56c6c'], ax=ax)
        ax.set_xlabel('是否患病')
        ax.set_ylabel('人数')
        ax.set_title('患病人数统计')
        ax.set_xticklabels(['未患病', '患病'])
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom')
        st.pyplot(fig)


# ============================
# 页面3：数据可视化
# ============================
def page_visualization(df):
    st.title("📈 数据可视化")
    st.markdown("---")

    viz_type = st.selectbox(
        "选择可视化类型",
        ["📊 各指标分布直方图",
         "🔥 相关性热力图",
         "⚖️ 患病 vs 未患病对比",
         "🎂 年龄分布与患病关系",
         "🕸️ 特征箱线图"]
    )

    # 各指标分布直方图
    if "直方图" in viz_type:
        st.subheader("📊 各指标分布直方图")

        feature_cols = [col for col in df.columns if col != 'Outcome']
        feature = st.selectbox("选择特征", feature_cols)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 整体分布
        sns.histplot(data=df, x=feature, kde=True, ax=axes[0], color='#409eff', bins=20)
        axes[0].set_title(f'{feature} 整体分布')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('频数')

        # 按患病分组分布
        sns.histplot(data=df, x=feature, hue='Outcome', kde=True, ax=axes[1],
                     palette=['#67c23a', '#f56c6c'], bins=20)
        axes[1].set_title(f'{feature} 按患病分布对比')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('频数')
        axes[1].legend(['患病', '未患病'])

        plt.tight_layout()
        st.pyplot(fig)

        # 说明
        if feature == 'Glucose':
            st.info("💡 **观察**：血糖（Glucose）是区分糖尿病最重要的指标之一，患病群体的血糖水平明显高于未患病群体。正常空腹血糖一般在 70-99 mg/dL 之间。")
        elif feature == 'BMI':
            st.info("💡 **观察**：BMI（体重指数）与糖尿病风险呈正相关，超重和肥胖人群的糖尿病患病率显著更高。中国成人BMI正常范围为18.5-23.9。")
        elif feature == 'Age':
            st.info("💡 **观察**：年龄越大，糖尿病患病率越高，这与临床上2型糖尿病多见于中老年人的规律一致。")

    # 相关性热力图
    elif "热力图" in viz_type:
        st.subheader("🔥 特征相关性热力图")

        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title('特征相关性热力图')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **分析说明**：
        - **Glucose（血糖）** 与 Outcome 的相关性最高（约 0.47），说明血糖是预测糖尿病最重要的指标
        - **BMI（体重指数）** 和 **Age（年龄）** 也与患病有一定相关性
        - **BloodPressure（血压）** 和 **SkinThickness（皮褶厚度）** 相关性较低
        - **Pregnancies（怀孕次数）** 和 **Age（年龄）** 有一定正相关，符合预期
        """)

    # 患病 vs 未患病对比
    elif "对比" in viz_type:
        st.subheader("⚖️ 患病与未患病特征对比")

        feature_cols = [col for col in df.columns if col != 'Outcome']

        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()

        for i, feature in enumerate(feature_cols):
            sns.boxplot(x='Outcome', y=feature, data=df, ax=axes[i],
                        palette=['#67c23a', '#f56c6c'])
            axes[i].set_title(f'{feature} 对比')
            axes[i].set_xlabel('是否患病')
            axes[i].set_xticklabels(['未患病', '患病'])

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **对比分析**：
        - 血糖（Glucose）：患病组中位数明显高于未患病组，差异最显著
        - BMI：患病组整体偏高
        - 年龄：患病组年龄普遍更大
        - 胰岛素（Insulin）：两组分布差异较大，但存在大量缺失值（0值）
        """)

    # 年龄分布与患病关系
    elif "年龄" in viz_type:
        st.subheader("🎂 年龄分布与患病关系")

        # 年龄分段
        df_copy = df.copy()
        df_copy['AgeGroup'] = pd.cut(df_copy['Age'],
                                     bins=[20, 30, 40, 50, 60, 100],
                                     labels=['20-29岁', '30-39岁', '40-49岁', '50-59岁', '60岁以上'])

        # 各年龄段患病率
        age_stats = df_copy.groupby('AgeGroup', observed=True)['Outcome'].agg(['count', 'mean'])
        age_stats.columns = ['总人数', '患病率']
        age_stats['患病数'] = (age_stats['总人数'] * age_stats['患病率']).astype(int)
        age_stats['患病率(%)'] = (age_stats['患病率'] * 100).round(2)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(age_stats[['总人数', '患病数', '患病率(%)']], use_container_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            age_stats['患病率(%)'].plot(kind='bar', ax=ax, color='#e6a23c', alpha=0.8)
            ax.set_xlabel('年龄段')
            ax.set_ylabel('患病率 (%)')
            ax.set_title('各年龄段糖尿病患病率')
            for i, v in enumerate(age_stats['患病率(%)']):
                ax.text(i, v + 0.5, f'{v}%', ha='center')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        st.info("""
        💡 **临床意义**：
        - 随着年龄增长，糖尿病患病率逐渐升高
        - 40岁以上人群患病率显著上升，这与2型糖尿病的发病规律一致
        - 2型糖尿病多在35~40岁之后发病，占糖尿病患者90%以上
        """)

    # 箱线图
    elif "箱线图" in viz_type:
        st.subheader("🕸️ 各特征箱线图")

        feature_cols = [col for col in df.columns if col != 'Outcome']
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()

        for i, feature in enumerate(feature_cols):
            sns.boxplot(y=feature, data=df, ax=axes[i], color='#409eff')
            axes[i].set_title(f'{feature} 箱线图')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **箱线图说明**：
        - 箱体代表四分位距（IQR），中间线为中位数
        - 上下须线表示数据范围
        - 须线外的点为异常值
        - 胰岛素（Insulin）和糖尿病谱系函数的离群值较多
        """)


# ============================
# 页面4：预测模型
# ============================
def page_model(df):
    st.title("🤖 糖尿病预测模型")
    st.markdown("---")

    # 训练模型
    model, scaler, metrics, feature_names = train_model(df)

    # 模型介绍
    st.subheader("📚 模型介绍")
    st.info("""
    **逻辑回归 (Logistic Regression)** 是一种经典的分类算法，虽然名字里有"回归"，
    但它主要用于二分类问题。它通过 Sigmoid 函数将线性回归的输出映射到 0-1 之间，
    表示样本属于正类的概率。

    **选择逻辑回归的原因**：
    - 模型简单，可解释性强，每个特征的权重有明确的医学意义
    - 训练速度快，适合小样本数据集
    - 在医疗诊断领域应用广泛，是临床预测模型的常用方法
    """)

    # 模型评估指标
    st.subheader("📊 模型评估")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("准确率 (Accuracy)", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("精确率 (Precision)", f"{metrics['precision']*100:.2f}%")
    col3.metric("召回率 (Recall)", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1 分数", f"{metrics['f1']*100:.2f}%")

    col5, col6 = st.columns(2)
    col5.metric("AUC 值", f"{metrics['roc_auc']:.4f}")
    col6.metric("训练集比例", "80% / 20% (测试集)")

    # 混淆矩阵和ROC曲线
    st.subheader("📈 可视化评估")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 混淆矩阵")
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['预测阴性', '预测阳性'],
                    yticklabels=['实际阴性', '实际阳性'])
        ax.set_title('混淆矩阵')
        ax.set_ylabel('真实标签')
        ax.set_xlabel('预测标签')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### ROC 曲线")
        fig, ax = plt.subplots(figsize=(6, 5))
        fpr, tpr = metrics['fpr'], metrics['tpr']
        roc_auc = metrics['roc_auc']
        ax.plot(fpr, tpr, color='#409eff', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (FPR)')
        ax.set_ylabel('真阳性率 (TPR)')
        ax.set_title('ROC 曲线')
        ax.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)

    # 指标说明
    with st.expander("📖 评估指标详解（点击展开）"):
        st.markdown("""
        **准确率 (Accuracy)**：预测正确的样本占总样本的比例
        $$Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}$$

        **精确率 (Precision)**：预测为阳性的样本中，实际为阳性的比例
        $$Precision = \\frac{TP}{TP + FP}$$
        *在医疗诊断中，精确率高意味着误诊率低*

        **召回率 (Recall / Sensitivity)**：实际为阳性的样本中，被正确预测的比例
        $$Recall = \\frac{TP}{TP + FN}$$
        *在医疗诊断中，召回率（灵敏度）高意味着漏诊率低，这对疾病筛查尤为重要*

        **F1 分数**：精确率和召回率的调和平均数
        $$F1 = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$$

        **AUC (Area Under ROC Curve)**：ROC曲线下的面积，取值0-1
        - AUC = 0.5：相当于随机猜测
        - AUC = 0.7-0.8：可接受
        - AUC = 0.8-0.9：良好
        - AUC > 0.9：优秀

        **混淆矩阵说明**：
        - TP (True Positive)：真阳性，正确预测为患病
        - TN (True Negative)：真阴性，正确预测为未患病
        - FP (False Positive)：假阳性，误诊（实际未患病但预测患病）
        - FN (False Negative)：假阴性，漏诊（实际患病但预测未患病）
        """)

    # 特征重要性
    st.subheader("🔍 特征重要性")
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': model.coef_[0],
        '绝对值': np.abs(model.coef_[0])
    }).sort_values('绝对值', ascending=False)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(coef_df[['特征', '系数']], use_container_width=True, hide_index=True)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#f56c6c' if c > 0 else '#67c23a' for c in coef_df['系数']]
        bars = ax.barh(coef_df['特征'], coef_df['系数'], color=colors)
        ax.set_xlabel('系数值')
        ax.set_title('逻辑回归特征系数（正=增加风险，负=降低风险）')
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("""
    **系数解读**：
    - **正系数**（红色）：特征值越大，糖尿病风险越高
    - **负系数**（绿色）：特征值越大，糖尿病风险越低
    - **系数绝对值越大**，对预测结果的影响越大
    - **Glucose（血糖）** 系数最大，再次验证了血糖是糖尿病最重要的预测指标
    """)


# ============================
# 页面5：风险计算器
# ============================
def page_calculator(df):
    st.title("🧮 糖尿病风险计算器")
    st.markdown("---")

    model, scaler, metrics, feature_names = train_model(df)

    st.info("""
    基于逻辑回归模型，输入以下指标，实时预测糖尿病患病概率。

    ⚠️ **免责声明**：本工具仅供学习和参考，不能替代专业医疗诊断。
    如有健康问题，请及时咨询专业医生。
    """)

    st.subheader("📝 请输入各项指标")

    # 初始化 session_state 中的默认值
    default_values = {
        'p_val': 2, 'g_val': 100, 'bp_val': 72, 'st_val': 25,
        'i_val': 80, 'b_val': 25.0, 'd_val': 0.4, 'a_val': 30
    }
    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("怀孕次数", min_value=0, max_value=20, key='p_val', step=1)
        glucose = st.number_input("血糖浓度 (mg/dL)", min_value=0, max_value=300, key='g_val', step=5)
        blood_pressure = st.number_input("舒张压 (mm Hg)", min_value=0, max_value=200, key='bp_val', step=2)
        skin_thickness = st.number_input("皮褶厚度 (mm)", min_value=0, max_value=100, key='st_val', step=1)

    with col2:
        insulin = st.number_input("胰岛素 (mu U/ml)", min_value=0, max_value=500, key='i_val', step=5)
        bmi = st.number_input("BMI 体重指数", min_value=0.0, max_value=60.0, key='b_val', step=0.1)
        dpf = st.number_input("糖尿病谱系函数", min_value=0.0, max_value=3.0, key='d_val', step=0.05)
        age = st.number_input("年龄 (岁)", min_value=21, max_value=100, key='a_val', step=1)

    if st.button("🔬 开始预测", type="primary", use_container_width=True):
        # 构造输入数据
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])

        # 标准化
        input_scaled = scaler.transform(input_data)

        # 预测
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        risk_prob = probability[1] * 100

        st.subheader("📊 预测结果")

        # 风险等级
        if risk_prob < 20:
            risk_level = "低风险"
            risk_color = "green"
            advice = "继续保持健康的生活方式，定期体检即可。"
        elif risk_prob < 50:
            risk_level = "中等风险"
            risk_color = "orange"
            advice = "建议注意饮食控制，增加运动，定期监测血糖。"
        elif risk_prob < 75:
            risk_level = "较高风险"
            risk_color = "red"
            advice = "建议尽快到医院进行进一步检查，注意控制饮食和体重。"
        else:
            risk_level = "高风险"
            risk_color = "red"
            advice = "患病概率很高，强烈建议尽快就医检查！"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("患病概率", f"{risk_prob:.1f}%")

        with col2:
            if risk_color == "green":
                st.success(f"风险等级：{risk_level}")
            elif risk_color == "orange":
                st.warning(f"风险等级：{risk_level}")
            else:
                st.error(f"风险等级：{risk_level}")

        with col3:
            st.metric("预测结果", "患病 ⚠️" if prediction == 1 else "未患病 ✅")

        # 进度条展示
        st.progress(risk_prob / 100)

        st.info(f"💡 健康建议：{advice}")

        # 各特征贡献
        with st.expander("📈 查看各特征对风险的贡献（点击展开）"):
            coef = model.coef_[0]
            input_scaled_flat = input_scaled[0]

            # 计算各特征的贡献值（标准化后的值 * 系数）
            contributions = input_scaled_flat * coef
            contrib_df = pd.DataFrame({
                '特征': feature_names,
                '输入值': input_data[0],
                '标准化后': input_scaled_flat,
                '特征系数': coef,
                '贡献值': contributions
            }).sort_values('贡献值', key=abs, ascending=False)

            st.dataframe(contrib_df.style.format({
                '输入值': '{:.1f}',
                '标准化后': '{:.3f}',
                '特征系数': '{:.3f}',
                '贡献值': '{:.3f}'
            }), use_container_width=True, hide_index=True)

            st.markdown("""
            **贡献值解读**：
            - 贡献值为正 → 增加患病风险
            - 贡献值为负 → 降低患病风险
            - 贡献值绝对值越大 → 对当前预测结果影响越大
            """)

    # 快速测试
    st.subheader("⚡ 快速测试示例")
    st.caption("点击下方按钮快速填充典型案例数据")

    col1, col2, col3 = st.columns(3)

    def set_example(p, g, bp, st_val, i, b, d, a):
        st.session_state.p_val = p
        st.session_state.g_val = g
        st.session_state.bp_val = bp
        st.session_state.st_val = st_val
        st.session_state.i_val = i
        st.session_state.b_val = b
        st.session_state.d_val = d
        st.session_state.a_val = a

    with col1:
        st.button("👩‍🦰 健康青年女性", on_click=set_example,
                  args=(0, 95, 68, 22, 70, 21.5, 0.25, 25))

    with col2:
        st.button("👩 中年肥胖高风险", on_click=set_example,
                  args=(4, 160, 90, 35, 180, 35.0, 0.6, 48))

    with col3:
        st.button("👵 老年糖尿病患者", on_click=set_example,
                  args=(8, 190, 95, 30, 130, 32.0, 0.8, 60))


# ============================
# 主函数
# ============================
def main():
    # 加载数据
    df = load_data()

    # 侧边栏导航
    page = sidebar()

    # 根据选择渲染对应页面
    if "数据集介绍" in page:
        page_intro(df)
    elif "数据概览" in page:
        page_overview(df)
    elif "数据可视化" in page:
        page_visualization(df)
    elif "预测模型" in page:
        page_model(df)
    elif "风险计算器" in page:
        page_calculator(df)


if __name__ == "__main__":
    main()

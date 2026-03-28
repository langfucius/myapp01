import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def _check_column_exists(df, column, func_name):
    """检查列是否存在"""
    if column not in df.columns:
        print(f"{func_name}: 列 '{column}' 不存在")
        return False
    return True


def _check_non_empty(data, func_name):
    """检查数据是否为空"""
    if data is None or (hasattr(data, 'empty') and data.empty) or (hasattr(data, '__len__') and len(data) == 0):
        print(f"{func_name}: 数据为空")
        return False
    return True


def plot_histogram(df, column):
    """
    绘制直方图
    """
    func_name = "plot_histogram"

    if not _check_column_exists(df, column, func_name):
        return None

    data = df[column].dropna()
    if not _check_non_empty(data, func_name):
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data, kde=True, ax=ax)
        ax.set_title(f"{column} 的分布直方图")
        ax.set_xlabel(column)
        ax.set_ylabel("频数")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_boxplot(df, column):
    """
    绘制箱线图
    """
    func_name = "plot_boxplot"

    if not _check_column_exists(df, column, func_name):
        return None

    data = df[column].dropna()
    if not _check_non_empty(data, func_name):
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=data, ax=ax)
        ax.set_title(f"{column} 的箱线图")
        ax.set_xlabel(column)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_bar_chart(df, column, n=10):
    """
    绘制条形图（类别计数）
    """
    func_name = "plot_bar_chart"

    if not _check_column_exists(df, column, func_name):
        return None

    value_counts = df[column].astype(str).value_counts().head(n)
    if not _check_non_empty(value_counts, func_name):
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
        ax.set_title(f"{column} 的前 {n} 个类别计数图")
        ax.set_xlabel("数量")
        ax.set_ylabel(column)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_scatter(df, x_col, y_col, hue_col=None):
    """
    绘制散点图
    """
    func_name = "plot_scatter"

    if not _check_column_exists(df, x_col, func_name):
        return None
    if not _check_column_exists(df, y_col, func_name):
        return None

    # 准备数据
    cols = [x_col, y_col]
    if hue_col and hue_col in df.columns:
        cols.append(hue_col)

    plot_df = df[cols].dropna()
    if not _check_non_empty(plot_df, func_name):
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        if hue_col and hue_col in df.columns:
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        else:
            sns.scatterplot(data=plot_df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"{x_col} vs {y_col} 散点图")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_line_chart(df, x_col, y_col):
    """
    绘制折线图
    """
    func_name = "plot_line_chart"

    if not _check_column_exists(df, x_col, func_name):
        return None
    if not _check_column_exists(df, y_col, func_name):
        return None

    plot_df = df[[x_col, y_col]].dropna().copy()
    if not _check_non_empty(plot_df, func_name):
        return None

    if len(plot_df) < 2:
        print(f"{func_name}: 数据点不足，需要至少2个点")
        return None

    # 确保 x 列可以排序
    try:
        plot_df = plot_df.sort_values(by=x_col)
    except:
        pass

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=plot_df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"{y_col} 随时间变化")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_grouped_boxplot(df, x_col, y_col):
    """
    绘制分组箱线图
    """
    func_name = "plot_grouped_boxplot"

    if not _check_column_exists(df, x_col, func_name):
        return None
    if not _check_column_exists(df, y_col, func_name):
        return None

    plot_df = df[[x_col, y_col]].dropna()
    if not _check_non_empty(plot_df, func_name):
        return None

    # 检查分组后是否有足够数据
    if plot_df[x_col].nunique() < 2:
        print(f"{func_name}: 分组变量 {x_col} 只有1个类别")
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"{y_col} 按 {x_col} 分组箱线图")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_correlation_heatmap(df, columns):
    """
    绘制相关性热力图
    """
    func_name = "plot_correlation_heatmap"

    if df.empty:
        print(f"{func_name}: DataFrame 为空")
        return None

    if not columns or len(columns) < 2:
        print(f"{func_name}: 需要至少2个列")
        return None

    # 检查存在的列
    existing = [c for c in columns if c in df.columns]
    if len(existing) < 2:
        print(f"{func_name}: 有效列不足2个，存在的列: {existing}")
        return None

    # 提取数据并删除缺失值
    temp = df[existing].dropna()
    if temp.empty:
        print(f"{func_name}: 删除缺失值后数据为空")
        return None

    if temp.shape[0] < 2:
        print(f"{func_name}: 数据行数不足2行")
        return None

    # 检查常数列
    constant_cols = []
    for col in existing:
        if temp[col].nunique() == 1:
            constant_cols.append(col)
            print(f"{func_name}: 警告: {col} 是常数列")

    if len(constant_cols) == len(existing):
        print(f"{func_name}: 所有列都是常数列，无法计算相关性")
        return None

    try:
        corr = temp.corr()

        if corr.empty or corr.isnull().all().all():
            print(f"{func_name}: 相关性矩阵全为 NaN")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                    square=True, cbar_kws={"shrink": 0.8})
        ax.set_title("相关性热力图")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_missing_values(df):
    """
    绘制缺失值条形图
    """
    func_name = "plot_missing_values"

    if df.empty:
        print(f"{func_name}: DataFrame 为空")
        return None

    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if missing_counts.empty:
        print(f"{func_name}: 没有缺失值")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, max(5, len(missing_counts) * 0.3)))
        sns.barplot(x=missing_counts.values, y=missing_counts.index, ax=ax)
        ax.set_title("各列缺失值数量")
        ax.set_xlabel("缺失值数量")
        ax.set_ylabel("列名")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_feature_importance(feature_importance_df, top_n=20):
    """
    绘制特征重要性条形图
    """
    func_name = "plot_feature_importance"

    if feature_importance_df is None or feature_importance_df.empty:
        print(f"{func_name}: 特征重要性数据为空")
        return None

    required_cols = ["特征", "重要性"]
    for col in required_cols:
        if col not in feature_importance_df.columns:
            print(f"{func_name}: 缺少列 '{col}'，当前列: {feature_importance_df.columns.tolist()}")
            return None

    top_df = feature_importance_df.head(top_n).copy()
    if top_df.empty:
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_df) * 0.3)))
        sns.barplot(data=top_df, x="重要性", y="特征", ax=ax)
        ax.set_title(f"前 {min(top_n, len(top_df))} 个重要特征")
        ax.set_xlabel("重要性")
        ax.set_ylabel("特征")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_roc_curve_binary(y_true, y_score, positive_label_name="正类"):
    """
    绘制二分类 ROC 曲线
    """
    func_name = "plot_roc_curve_binary"

    # 转换为 numpy 数组
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # 删除 NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        print(f"{func_name}: 数据为空")
        return None

    # 检查类别数量
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print(f"{func_name}: 只有1个类别 {unique_classes}，无法绘制 ROC 曲线")
        return None

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC 曲线（{positive_label_name}）")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_regression_actual_vs_pred(y_true, y_pred):
    """
    绘制回归真实值 vs 预测值散点图
    """
    func_name = "plot_regression_actual_vs_pred"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        print(f"{func_name}: 长度不一致，真实值 {len(y_true)}，预测值 {len(y_pred)}")
        return None

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"{func_name}: 数据为空")
        return None

    try:
        plot_df = pd.DataFrame({"真实值": y_true, "预测值": y_pred})

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(data=plot_df, x="真实值", y="预测值", ax=ax, alpha=0.6)

        min_val = min(plot_df["真实值"].min(), plot_df["预测值"].min())
        max_val = max(plot_df["真实值"].max(), plot_df["预测值"].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", linewidth=2)

        ax.set_title("真实值 vs 预测值")
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_regression_residuals(y_true, y_pred):
    """
    绘制回归残差图
    """
    func_name = "plot_regression_residuals"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        print(f"{func_name}: 长度不一致")
        return None

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        print(f"{func_name}: 数据为空")
        return None

    residuals = y_true - y_pred

    try:
        plot_df = pd.DataFrame({"预测值": y_pred, "残差": residuals})

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=plot_df, x="预测值", y="残差", ax=ax, alpha=0.6)
        ax.axhline(0, linestyle="--", color="red", linewidth=2)
        ax.set_title("残差图")
        ax.set_xlabel("预测值")
        ax.set_ylabel("残差")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None


def plot_pca_clusters(X_scaled, labels):
    """
    绘制 PCA 二维聚类可视化图
    """
    func_name = "plot_pca_clusters"

    if X_scaled is None:
        print(f"{func_name}: X_scaled 为空")
        return None

    if len(X_scaled) < 2:
        print(f"{func_name}: 样本数不足2")
        return None

    if len(X_scaled) != len(labels):
        print(f"{func_name}: 样本数与标签数不一致")
        return None

    # 转换为 numpy 并删除 NaN
    X_scaled = np.array(X_scaled)
    labels = np.array(labels)

    mask = ~(np.isnan(X_scaled).any(axis=1))
    X_scaled = X_scaled[mask]
    labels = labels[mask]

    if len(X_scaled) < 2:
        print(f"{func_name}: 删除 NaN 后样本数不足2")
        return None

    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plot_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": labels.astype(str)
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster",
                        palette="tab10", ax=ax, alpha=0.7)
        ax.set_title(
            f"PCA 二维聚类可视化\n(解释方差: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%})")
        ax.set_xlabel("第一主成分")
        ax.set_ylabel("第二主成分")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"{func_name} 绘图失败: {e}")
        return None
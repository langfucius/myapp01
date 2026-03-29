import streamlit as st
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_path = "SourceHanSansSC-Regular.otf"

if not os.path.exists(font_path):
    import urllib.request
    urllib.request.urlretrieve(font_url, font_path)

fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Source Han Sans SC'
plt.rcParams['axes.unicode_minus'] = False

from datetime import datetime
import pytz
from utils.data_loader import load_data_csv, load_data_excel
from utils.data_cleaning import (
    clean_data,
    get_missing_value_summary,
    get_cleaning_summary,
    get_dtype_summary
)
from utils.visualization import (
    plot_histogram,
    plot_boxplot,
    plot_bar_chart,
    plot_scatter,
    plot_line_chart,
    plot_grouped_boxplot,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_feature_importance,
    plot_roc_curve_binary,
    plot_regression_actual_vs_pred,
    plot_regression_residuals,
    plot_pca_clusters
)
from utils.model_training import (
    train_classification_model,
    train_regression_model,
    run_kmeans_clustering,
    calculate_elbow_method
)
from utils.model_utils import (
    serialize_model_to_bytes,
    predict_with_trained_model,
    predict_with_trained_classification_model
)

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import requests
# ========== 获取公网IP和地理位置 ==========
def get_public_ip():
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", "")
        if ip and ip != "unknown":
            first_ip = ip.split(",")[0].strip()
            if not first_ip.startswith(("192.168.", "10.", "172.16.", "127.")):
                return first_ip
        response = requests.get("https://api.ipify.org", timeout=3)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return "unknown"

def get_ip_location(ip):
    if ip == "unknown" or ip.startswith(("192.168.", "10.", "172.", "127.")):
        return {"country": "内网IP", "city": "无法定位", "isp": "局域网"}
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}?fields=status,country,city,isp", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return {
                    "country": data.get("country", "未知"),
                    "city": data.get("city", "未知"),
                    "isp": data.get("isp", "未知")
                }
    except:
        pass
    return {"country": "定位失败", "city": "定位失败", "isp": "定位失败"}

# ========== 记录普通访客 ==========
def log_visitor():
    """记录普通访客（不需要密码）"""
    try:
        public_ip = get_public_ip()
        location = get_ip_location(public_ip)
        
        visitor_info = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "普通访客",
            "public_ip": public_ip,
            "country": location["country"],
            "city": location["city"],
            "isp": location["isp"],
            "device": st.context.headers.get("User-Agent", "unknown")[:150]
        }
        
        log_file = "visitors.csv"
        df_new = pd.DataFrame([visitor_info])
        
        if os.path.exists(log_file):
            df_old = pd.read_csv(log_file)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        
        df_new.to_csv(log_file, index=False)
    except:
        pass

# ========== 记录管理员登录 ==========
def log_admin(level):
    """记录管理员登录"""
    try:
        public_ip = get_public_ip()
        location = get_ip_location(public_ip)
        
        admin_info = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": f"管理员(密钥{level})",
            "public_ip": public_ip,
            "country": location["country"],
            "city": location["city"],
            "isp": location["isp"],
            "device": st.context.headers.get("User-Agent", "unknown")[:150]
        }
        
        log_file = "admin_logins.csv"  # 单独记录管理员登录
        df_new = pd.DataFrame([admin_info])
        
        if os.path.exists(log_file):
            df_old = pd.read_csv(log_file)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        
        df_new.to_csv(log_file, index=False)
    except:
        pass

# ========== 普通访客自动记录 ==========
if "visitor_logged" not in st.session_state:
    st.session_state.visitor_logged = True
    log_visitor()

# ========== 管理员登录入口（侧边栏） ==========
KEY_A = "1117"   # 密钥A：普通管理权限（只能看普通访客）
KEY_B = "050508"   # 密钥B：高级管理权限（能看普通访客 + 所有管理员登录记录）

if "admin_level" not in st.session_state:
    st.session_state.admin_level = None

with st.sidebar:
    st.markdown("---")
    if st.session_state.admin_level is None:
        st.markdown("### 🔐 管理员登录")
        pwd = st.text_input("请输入密钥", type="password", key="admin_pwd")
        if st.button("登录", key="admin_login_btn"):
            if pwd == KEY_A:
                st.session_state.admin_level = "A"
                log_admin("A")
                st.rerun()
            elif pwd == KEY_B:
                st.session_state.admin_level = "B"
                log_admin("B")
                st.rerun()
            elif pwd:
                st.error("密钥错误")
    else:
        st.success(f"管理员(密钥)已登录")
        if st.button("🚪 退出管理"):
            st.session_state.admin_level = None
            st.rerun()

# ========== 管理员面板 ==========
if st.session_state.admin_level is not None:
    with st.sidebar:
        st.markdown("### 📊 管理面板")
        
        # 查看所有普通访客记录
        if st.button("👥 查看普通访客记录"):
            if os.path.exists("visitors.csv"):
                df = pd.read_csv("visitors.csv")
                st.dataframe(df)
                st.info(f"总访客：{len(df)} 次")
        
        # 密钥B 还能看管理员登录记录
        if st.session_state.admin_level == "B":
            st.markdown("---")
            
            if st.button("🔑 查看所有管理员登录记录"):
                if os.path.exists("admin_logins.csv"):
                    df = pd.read_csv("admin_logins.csv")
                    st.dataframe(df)
                    st.info(f"管理员登录次数：{len(df)} 次")
                    
                    # 统计密钥A和密钥B的使用情况
                    level_stats = df.groupby("type").size().reset_index(name="次数")
                    st.markdown("### 密钥使用统计")
                    st.dataframe(level_stats)
            
            if st.button("📍 位置分布"):
                if os.path.exists("visitors.csv"):
                    df = pd.read_csv("visitors.csv")
                    location_stats = df.groupby(["country", "city"]).size().reset_index(name="次数")
                    st.dataframe(location_stats)
            
            if st.button("💻 设备统计"):
                if os.path.exists("visitors.csv"):
                    df = pd.read_csv("visitors.csv")
                    device_stats = df.groupby("device").size().reset_index(name="次数").head(20)
                    st.dataframe(device_stats)
            
            st.markdown("---")
            if st.button("📥 下载所有数据"):
                if os.path.exists("visitors.csv"):
                    df = pd.read_csv("visitors.csv")
                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("下载普通访客记录.csv", csv, "visitors.csv", "text/csv")
                
                if os.path.exists("admin_logins.csv"):
                    df_admin = pd.read_csv("admin_logins.csv")
                    csv_admin = df_admin.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("下载管理员登录记录.csv", csv_admin, "admin_logins.csv", "text/csv")

# ========== 下面是你原来的代码 ==========
# ... 你的 main() 函数和所有其他代码 ...
# ====================== 路径处理（支持打包） ======================
def get_base_path():
    """获取应用基础路径（支持打包后）"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


# ====================== 工具函数 ======================
def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def load_uploaded_data(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return load_data_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return load_data_excel(uploaded_file)
    else:
        return None, "不支持的文件格式"


def build_type_config_ui(df):
    st.markdown("### 数据类型转换设置（可选）")
    type_config = {}

    cols = st.columns(2)
    for idx, col in enumerate(df.columns):
        with cols[idx % 2]:
            selected_type = st.selectbox(
                f"{col} 类型",
                ["保持不变", "numeric", "category", "datetime", "string"],
                key=f"type_convert_{col}"
            )
            if selected_type != "保持不变":
                type_config[col] = selected_type

    return type_config


# ====================== 主函数 ======================
def main():
    """所有 Streamlit UI 代码放在这个函数里"""

    # 设置 matplotlib 字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 页面配置
    st.set_page_config(page_title="Mini ML App", layout="wide")


# 北京时间5月17日祝福


    st.title("Mini ML Lab")
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    if now.month == 5 and now.day == 17:
        st.balloons()
        
        # 播放本地生日歌
        with open("birthday.mp3", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=True)
        
        st.success("5月17日生日快乐！")
        st.info("愿星辰指引你的方向，愿未来如你所愿，二十岁生日快乐呀!-- 献给User 0")
        st.info("I will come.")
        st.snow()
    st.write("The best ot data times, the worst of data times.--Tinpot author")
    st.write("The stars must be aligned tonight.--User 0")
    st.write("星光今夜交相辉映。——用户0")
    st.write("Las estrellas deben estar alineadas esta noche.--Usuario 0")
    st.write("0يجب أن تكون النجوم متحاذية الليلة.--المستخدم")
    st.write("Звёзды должны выровняться этой ночью.--Пользователь 0")
    st.write("Les étoiles doivent être alignées ce soir.--Utilisateur 0")
    
    st.write("Thanks for MYF and the lazy Lang")

    # ======================
    # Session 初始化
    # ======================
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None

    if "trained_model_type" not in st.session_state:
        st.session_state["trained_model_type"] = None

    if "trained_target_column" not in st.session_state:
        st.session_state["trained_target_column"] = None

    if "trained_feature_columns" not in st.session_state:
        st.session_state["trained_feature_columns"] = None

    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None

    if "classification_label_encoder" not in st.session_state:
        st.session_state["classification_label_encoder"] = None

    # ======================
    # 侧边栏
    # ======================
    st.sidebar.title("功能导航")
    page = st.sidebar.radio(
        "请选择功能",
        ["首页", "数据预览", "数据清洗", "数据可视化", "分类模型", "回归模型", "聚类分析", "模型预测"],
        key="page_radio"
    )

    uploaded_file = st.file_uploader(
        "请上传一个文件",
        type=["csv", "xlsx", "xls"],
        key="file_uploader"
    )

    # ======================
    # 首页
    # ======================
    if page == "首页":
        st.subheader("WELCOME")
        st.write("这是一个用于数据分析、可视化和机器学习的工具。")

        st.markdown("### 当前支持功能")
        st.markdown("""
        - 数据预览
        - 数据清洗
        - 数据可视化
        - 分类模型训练
        - 回归模型训练
        - KMeans 聚类分析
        - 模型保存下载
        - 新数据预测
        - 交叉验证
        - GridSearchCV 超参数调优
        """)

        if uploaded_file is not None:
            st.success(f"已上传文件：{uploaded_file.name}")
            st.write(f"文件大小：{format_file_size(uploaded_file.size)}")
        else:
            st.info("请先上传 CSV 或 Excel 文件。")

        if st.session_state["trained_model"] is not None:
            st.markdown("### 当前内存中的模型")
            st.success(f"已存在已训练模型，类型：{st.session_state['trained_model_type']}")
            st.write(f"目标列：{st.session_state['trained_target_column']}")
            st.write(f"特征列：{st.session_state['trained_feature_columns']}")

    # ======================
    # 其他页面需要上传文件
    # ======================
    else:
        if uploaded_file is None and page != "模型预测":
            st.warning("请先上传 CSV 或 Excel 文件，再使用该功能。")
        else:
            df = None
            error = None

            if uploaded_file is not None:
                st.subheader("文件信息")
                st.write(f"文件名: {uploaded_file.name}")
                st.write(f"文件类型: {uploaded_file.type}")
                st.write(f"文件大小: {format_file_size(uploaded_file.size)}")

                df, error = load_uploaded_data(uploaded_file)

                if error:
                    st.error("文件读取失败，请检查文件格式。")
                    st.code(error)
                else:
                    st.success("文件读取成功！")

            # ======================
            # 模型预测页不强依赖主上传文件
            # ======================
            if page == "模型预测":
                st.subheader("模型预测")
                st.info("请先在“分类模型”或“回归模型”页面训练模型，然后再上传新数据进行预测。")

                if st.session_state["trained_model"] is None:
                    st.warning("当前没有已训练模型，请先到分类模型或回归模型页面训练模型。")
                else:
                    st.success(f"当前可用模型类型：{st.session_state['trained_model_type']}")
                    st.write(f"目标列：{st.session_state['trained_target_column']}")
                    st.write(f"训练时特征列：{st.session_state['trained_feature_columns']}")

                    predict_file = st.file_uploader(
                        "上传用于预测的新数据文件",
                        type=["csv", "xlsx", "xls"],
                        key="predict_file_uploader"
                    )

                    if predict_file is not None:
                        new_df, pred_error = load_uploaded_data(predict_file)

                        if pred_error:
                            st.error(pred_error)
                        else:
                            st.success("预测数据读取成功！")
                            st.dataframe(new_df.head(), use_container_width=True)

                            if st.button("开始预测", key="predict_btn"):
                                model = st.session_state["trained_model"]
                                feature_columns = st.session_state["trained_feature_columns"]
                                model_type = st.session_state["trained_model_type"]

                                missing_cols = [col for col in feature_columns if col not in new_df.columns]
                                if missing_cols:
                                    st.error(f"预测数据缺少以下特征列：{missing_cols}")
                                else:
                                    predict_input_df = new_df[feature_columns].copy()

                                    if model_type == "classification":
                                        result_df, pred_error = predict_with_trained_classification_model(
                                            model,
                                            predict_input_df,
                                            label_encoder=st.session_state.get("classification_label_encoder")
                                        )
                                    else:
                                        result_df, pred_error = predict_with_trained_model(model, predict_input_df)

                                    if pred_error:
                                        st.error(pred_error)
                                    else:
                                        st.success("预测完成！")
                                        st.dataframe(result_df.head(), use_container_width=True)

                                        pred_csv = result_df.to_csv(index=False).encode("utf-8-sig")
                                        st.download_button(
                                            label="下载预测结果 CSV",
                                            data=pred_csv,
                                            file_name="prediction_result.csv",
                                            mime="text/csv",
                                            key="download_prediction_csv"
                                        )

            elif error is None:
                # ======================
                # 侧边栏：数据清洗设置
                # ======================
                st.sidebar.markdown("---")
                st.sidebar.subheader("数据清洗设置")

                selected_columns = st.sidebar.multiselect(
                    "请选择要保留的列",
                    df.columns.tolist(),
                    default=df.columns.tolist(),
                    key="selected_columns_sidebar"
                )

                numeric_missing_method = st.sidebar.selectbox(
                    "数值列缺失值处理方式",
                    ["不处理", "删除缺失行", "均值填充", "中位数填充", "0填充", "线性插值", "前向填充", "后向填充"],
                    index=3,
                    key="numeric_missing_method"
                )

                categorical_missing_method = st.sidebar.selectbox(
                    "类别列缺失值处理方式",
                    ["不处理", "删除缺失行", "众数填充", "Unknown填充"],
                    index=2,
                    key="categorical_missing_method"
                )

                outlier_method = st.sidebar.selectbox(
                    "异常值处理方式",
                    ["不处理", "IQR剔除", "Z-score剔除"],
                    index=0,
                    key="outlier_method"
                )

                drop_duplicates = st.sidebar.checkbox(
                    "删除重复值",
                    key="drop_duplicates_sidebar"
                )

                if selected_columns:
                    with st.expander("数据类型转换设置", expanded=False):
                        type_config = build_type_config_ui(df[selected_columns])

                    df_cleaned = clean_data(
                        df,
                        selected_columns,
                        drop_duplicates=drop_duplicates,
                        numeric_missing_method=numeric_missing_method,
                        categorical_missing_method=categorical_missing_method,
                        outlier_method=outlier_method,
                        type_config=type_config
                    )
                else:
                    df_cleaned = pd.DataFrame()

                # ======================
                # 数据预览
                # ======================
                if page == "数据预览":
                    st.subheader("数据预览")

                    preview_rows = st.selectbox(
                        "选择预览行数",
                        [5, 10, 20, 50],
                        index=0,
                        key="preview_rows_select"
                    )
                    st.dataframe(df.head(preview_rows), use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("总行数", df.shape[0])
                    col2.metric("总列数", df.shape[1])
                    col3.metric("重复行数", int(df.duplicated().sum()))

                    st.subheader("每列信息统计")
                    col_info = df.dtypes.reset_index()
                    col_info.columns = ["列名", "数据类型"]
                    col_info["非空数量"] = df.notnull().sum().values
                    col_info["缺失值数量"] = df.isnull().sum().values
                    col_info["唯一值数量"] = df.nunique().values
                    st.dataframe(col_info, use_container_width=True)

                    st.subheader("缺失值统计")
                    missing_fig = plot_missing_values(df)
                    if missing_fig:
                        st.pyplot(missing_fig)
                    else:
                        st.success("当前数据没有缺失值。")

                    st.subheader("数值列描述统计")
                    numeric_df = df.select_dtypes(include=["number"])
                    if not numeric_df.empty:
                        st.dataframe(numeric_df.describe(), use_container_width=True)
                    else:
                        st.info("当前数据中没有数值列。")

                # ======================
                # 数据清洗
                # ======================
                elif page == "数据清洗":
                    st.subheader("数据清洗")

                    if df_cleaned.empty:
                        st.warning("请至少选择一列。")
                    else:
                        st.markdown("### 清洗策略")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"数值列缺失处理：{numeric_missing_method}")
                            st.info(f"类别列缺失处理：{categorical_missing_method}")
                        with col2:
                            st.info(f"异常值处理：{outlier_method}")
                            st.info(f"删除重复值：{'是' if drop_duplicates else '否'}")

                        st.markdown("### 清洗前后概览")
                        cleaning_summary = get_cleaning_summary(df[selected_columns], df_cleaned)
                        summary_df = pd.DataFrame({
                            "指标": list(cleaning_summary.keys()),
                            "值": list(cleaning_summary.values())
                        })
                        st.dataframe(summary_df, use_container_width=True)

                        st.markdown("### 清洗前缺失值统计")
                        st.dataframe(get_missing_value_summary(df[selected_columns]), use_container_width=True)

                        st.markdown("### 清洗后缺失值统计")
                        st.dataframe(get_missing_value_summary(df_cleaned), use_container_width=True)

                        st.markdown("### 清洗后数据类型")
                        st.dataframe(get_dtype_summary(df_cleaned), use_container_width=True)

                        st.markdown("### 清洗后数据预览")
                        st.dataframe(df_cleaned.head(), use_container_width=True)

                        csv_data = df_cleaned.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="下载清洗后的 CSV 文件",
                            data=csv_data,
                            file_name="cleaned_data.csv",
                            mime="text/csv",
                            key="download_cleaned_csv"
                        )

                # ======================
                # 数据可视化
                # ======================
                elif page == "数据可视化":
                    st.subheader("数据可视化")

                    if df_cleaned.empty:
                        st.warning("请先选择至少一列数据。")
                    else:
                        numeric_columns = df_cleaned.select_dtypes(include=["number"]).columns.tolist()
                        categorical_columns = df_cleaned.select_dtypes(
                            include=["object", "category", "bool"]).columns.tolist()

                        tab1, tab2, tab3, tab4 = st.tabs(["单变量分析", "双变量分析", "相关性分析", "缺失值分析"])

                        with tab1:
                            st.markdown("### 直方图")
                            if numeric_columns:
                                hist_col = st.selectbox("选择数值列", numeric_columns, key="hist_col")
                                st.pyplot(plot_histogram(df_cleaned, hist_col))

                                st.markdown("### 箱线图")
                                box_col = st.selectbox("选择箱线图数值列", numeric_columns, key="box_col")
                                st.pyplot(plot_boxplot(df_cleaned, box_col))
                            else:
                                st.info("没有数值列。")

                            st.markdown("### 类别计数图")
                            if categorical_columns:
                                bar_col = st.selectbox("选择类别列", categorical_columns, key="bar_col")
                                top_n = st.number_input(
                                    "显示前多少类",
                                    min_value=1,
                                    max_value=max(1, df_cleaned[bar_col].nunique()),
                                    value=min(10, max(1, df_cleaned[bar_col].nunique())),
                                    key="bar_top_n"
                                )
                                st.pyplot(plot_bar_chart(df_cleaned, bar_col, int(top_n)))
                            else:
                                st.info("没有类别列。")

                        with tab2:
                            st.markdown("### 散点图")
                            if len(numeric_columns) >= 2:
                                scatter_x = st.selectbox("X 轴", numeric_columns, key="scatter_x")
                                scatter_y = st.selectbox("Y 轴", numeric_columns, key="scatter_y")
                                hue_options = ["无"] + categorical_columns
                                scatter_hue = st.selectbox("颜色分组", hue_options, key="scatter_hue")
                                hue_val = None if scatter_hue == "无" else scatter_hue
                                st.pyplot(plot_scatter(df_cleaned, scatter_x, scatter_y, hue_val))
                            else:
                                st.info("至少需要两个数值列才能画散点图。")

                            st.markdown("### 折线图")
                            if len(df_cleaned.columns) >= 2 and numeric_columns:
                                line_x = st.selectbox("折线图 X 轴", df_cleaned.columns.tolist(), key="line_x")
                                line_y = st.selectbox("折线图 Y 轴（数值列）", numeric_columns, key="line_y")
                                st.pyplot(plot_line_chart(df_cleaned, line_x, line_y))
                            else:
                                st.info("折线图至少需要一个数值列。")

                            st.markdown("### 分组箱线图")
                            if categorical_columns and numeric_columns:
                                group_x = st.selectbox("分组类别列", categorical_columns, key="group_x")
                                group_y = st.selectbox("箱线图数值列", numeric_columns, key="group_y")
                                st.pyplot(plot_grouped_boxplot(df_cleaned, group_x, group_y))
                            else:
                                st.info("需要至少一个类别列和一个数值列。")

                        with tab3:
                            st.markdown("### 相关性热力图")
                            if len(numeric_columns) >= 2:
                                heatmap_cols = st.multiselect(
                                    "选择数值列（至少2列）",
                                    numeric_columns,
                                    default=numeric_columns[:min(5, len(numeric_columns))],
                                    key="heatmap_cols"
                                )
                                if len(heatmap_cols) >= 2:
                                    fig = plot_correlation_heatmap(df_cleaned, heatmap_cols)
                                    if fig:
                                        st.pyplot(fig)
                                else:
                                    st.warning("请至少选择 2 个数值列。")
                            else:
                                st.info("至少需要 2 个数值列。")

                        with tab4:
                            st.markdown("### 缺失值可视化")
                            fig = plot_missing_values(df_cleaned)
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.success("当前清洗后数据没有缺失值。")

                # ======================
                # 分类模型
                # ======================
                elif page == "分类模型":
                    st.subheader("机器学习：分类模型")

                    if df_cleaned.empty:
                        st.warning("请先选择至少一列数据。")
                    elif df_cleaned.shape[1] < 2:
                        st.info("至少需要两列数据，才能进行分类任务。")
                    else:
                        target_column = st.selectbox(
                            "请选择目标列（要预测的列）",
                            df_cleaned.columns.tolist(),
                            key="classification_target_column"
                        )

                        model_name = st.selectbox(
                            "请选择分类模型",
                            ["随机森林", "逻辑回归", "决策树", "KNN"],
                            key="classification_model_name"
                        )

                        test_size = st.slider(
                            "测试集比例",
                            min_value=0.1,
                            max_value=0.4,
                            value=0.2,
                            step=0.05,
                            key="classification_test_size"
                        )

                        use_cv = st.checkbox("启用交叉验证", key="classification_use_cv")
                        cv_folds = st.slider("交叉验证折数", 3, 10, 5, key="classification_cv_folds")
                        use_tuning = st.checkbox("启用超参数调优（GridSearchCV）", key="classification_use_tuning")

                        if st.button("开始训练分类模型", key="train_classification_btn"):
                            with st.spinner("模型训练中，请稍候..."):
                                result, train_error = train_classification_model(
                                    df_cleaned,
                                    target_column=target_column,
                                    model_name=model_name,
                                    test_size=test_size,
                                    random_state=42,
                                    use_cv=use_cv,
                                    cv_folds=cv_folds,
                                    use_tuning=use_tuning
                                )

                            if train_error:
                                st.error(train_error)
                            else:
                                st.session_state["trained_model"] = result["pipeline"]
                                st.session_state["trained_model_type"] = result["task_type"]
                                st.session_state["trained_target_column"] = result["target_column"]
                                st.session_state["trained_feature_columns"] = result["feature_columns"]
                                st.session_state["latest_result"] = result
                                st.session_state["classification_label_encoder"] = result["label_encoder"]

                                st.success("分类模型训练完成！")

                                if result["cv_scores"] is not None:
                                    st.markdown("### 交叉验证结果")
                                    cv_df = pd.DataFrame({
                                        "Fold": list(range(1, len(result["cv_scores"]) + 1)),
                                        "Accuracy": result["cv_scores"]
                                    })
                                    st.dataframe(cv_df, use_container_width=True)
                                    st.info(f"CV Mean = {result['cv_mean']:.4f}, CV Std = {result['cv_std']:.4f}")

                                if result["best_params"] is not None:
                                    st.markdown("### 超参数调优结果")
                                    st.json(result["best_params"])
                                    st.info(f"最佳交叉验证分数 = {result['best_cv_score']:.4f}")

                                col1, col2, col3 = st.columns(3)
                                col1.metric("准确率", f"{result['accuracy']:.4f}")
                                col2.metric("训练集大小", str(result["train_shape"]))
                                col3.metric("测试集大小", str(result["test_shape"]))

                                st.markdown("### 分类报告")
                                st.text(result["classification_report"])

                                st.markdown("### 真实值 vs 预测值")
                                st.dataframe(result["results_df"], use_container_width=True)

                                pred_csv = result["results_df"].to_csv(index=False).encode("utf-8-sig")
                                st.download_button(
                                    label="下载分类预测结果 CSV",
                                    data=pred_csv,
                                    file_name="classification_predictions.csv",
                                    mime="text/csv",
                                    key="download_classification_csv"
                                )

                                st.markdown("### 混淆矩阵")
                                cm = result["confusion_matrix"]
                                labels = result["label_classes"]

                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.heatmap(
                                    cm,
                                    annot=True,
                                    fmt="d",
                                    cmap="Blues",
                                    xticklabels=labels,
                                    yticklabels=labels,
                                    ax=ax
                                )
                                ax.set_xlabel("预测值")
                                ax.set_ylabel("真实值")
                                ax.set_title("混淆矩阵")
                                st.pyplot(fig)

                                st.markdown("### ROC 曲线")
                                if result["roc_available"]:
                                    roc_fig = plot_roc_curve_binary(
                                        result["y_test_encoded"],
                                        result["y_score"]
                                    )
                                    st.pyplot(roc_fig)
                                    st.info(f"AUC = {result['roc_auc']:.4f}")
                                else:
                                    st.info("当前模型/数据不满足二分类 ROC 绘制条件。")

                                st.markdown("### 特征重要性分析")
                                if result["feature_importance_df"] is not None and not result[
                                    "feature_importance_df"].empty:
                                    max_features = min(30, len(result["feature_importance_df"]))
                                    min_features = 1 if max_features < 5 else 5
                                    default_features = min(10, max_features)

                                    top_n_features = st.slider(
                                        "显示前多少个重要特征",
                                        min_value=min_features,
                                        max_value=max_features,
                                        value=default_features,
                                        step=1,
                                        key="classification_top_n_features"
                                    )
                                    st.dataframe(
                                        result["feature_importance_df"].head(top_n_features),
                                        use_container_width=True
                                    )
                                    st.pyplot(plot_feature_importance(result["feature_importance_df"], top_n_features))
                                else:
                                    st.info(result["importance_message"])

                                model_bytes = serialize_model_to_bytes(result["pipeline"])
                                st.download_button(
                                    label="下载分类模型文件 (.pkl)",
                                    data=model_bytes,
                                    file_name="classification_model.pkl",
                                    mime="application/octet-stream",
                                    key="download_classification_model"
                                )

                # ======================
                # 回归模型
                # ======================
                elif page == "回归模型":
                    st.subheader("机器学习：回归模型")

                    if df_cleaned.empty:
                        st.warning("请先选择至少一列数据。")
                    elif df_cleaned.shape[1] < 2:
                        st.info("至少需要两列数据，才能进行回归任务。")
                    else:
                        numeric_targets = df_cleaned.select_dtypes(include=["number"]).columns.tolist()

                        if not numeric_targets:
                            st.warning("当前清洗后数据中没有数值列，无法进行回归。")
                        else:
                            target_column = st.selectbox(
                                "请选择目标列（数值型）",
                                numeric_targets,
                                key="regression_target_column"
                            )

                            model_name = st.selectbox(
                                "请选择回归模型",
                                ["线性回归", "随机森林回归", "决策树回归", "KNN回归"],
                                key="regression_model_name"
                            )

                            test_size = st.slider(
                                "测试集比例",
                                min_value=0.1,
                                max_value=0.4,
                                value=0.2,
                                step=0.05,
                                key="regression_test_size"
                            )

                            use_cv = st.checkbox("启用交叉验证", key="regression_use_cv")
                            cv_folds = st.slider("交叉验证折数", 3, 10, 5, key="regression_cv_folds")
                            use_tuning = st.checkbox("启用超参数调优（GridSearchCV）", key="regression_use_tuning")

                            if st.button("开始训练回归模型", key="train_regression_btn"):
                                with st.spinner("回归模型训练中，请稍候..."):
                                    result, train_error = train_regression_model(
                                        df_cleaned,
                                        target_column=target_column,
                                        model_name=model_name,
                                        test_size=test_size,
                                        random_state=42,
                                        use_cv=use_cv,
                                        cv_folds=cv_folds,
                                        use_tuning=use_tuning
                                    )

                                if train_error:
                                    st.error(train_error)
                                else:
                                    st.session_state["trained_model"] = result["pipeline"]
                                    st.session_state["trained_model_type"] = result["task_type"]
                                    st.session_state["trained_target_column"] = result["target_column"]
                                    st.session_state["trained_feature_columns"] = result["feature_columns"]
                                    st.session_state["latest_result"] = result
                                    st.session_state["classification_label_encoder"] = None

                                    st.success("回归模型训练完成！")

                                    if result["cv_scores"] is not None:
                                        st.markdown("### 交叉验证结果")
                                        cv_df = pd.DataFrame({
                                            "Fold": list(range(1, len(result["cv_scores"]) + 1)),
                                            "R2": result["cv_scores"]
                                        })
                                        st.dataframe(cv_df, use_container_width=True)
                                        st.info(f"CV Mean = {result['cv_mean']:.4f}, CV Std = {result['cv_std']:.4f}")

                                    if result["best_params"] is not None:
                                        st.markdown("### 超参数调优结果")
                                        st.json(result["best_params"])
                                        st.info(f"最佳交叉验证分数 = {result['best_cv_score']:.4f}")

                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("MAE", f"{result['mae']:.4f}")
                                    col2.metric("MSE", f"{result['mse']:.4f}")
                                    col3.metric("RMSE", f"{result['rmse']:.4f}")
                                    col4.metric("R²", f"{result['r2']:.4f}")

                                    st.markdown("### 预测结果预览")
                                    st.dataframe(result["results_df"], use_container_width=True)

                                    pred_csv = result["results_df"].to_csv(index=False).encode("utf-8-sig")
                                    st.download_button(
                                        label="下载回归预测结果 CSV",
                                        data=pred_csv,
                                        file_name="regression_predictions.csv",
                                        mime="text/csv",
                                        key="download_regression_csv"
                                    )

                                    st.markdown("### 真实值 vs 预测值")
                                    st.pyplot(plot_regression_actual_vs_pred(result["y_test"], result["y_pred"]))

                                    st.markdown("### 残差图")
                                    st.pyplot(plot_regression_residuals(result["y_test"], result["y_pred"]))

                                    st.markdown("### 特征重要性分析")
                                    if result["feature_importance_df"] is not None and not result[
                                        "feature_importance_df"].empty:
                                        max_features = min(30, len(result["feature_importance_df"]))
                                        min_features = 1 if max_features < 5 else 5
                                        default_features = min(10, max_features)

                                        top_n_features = st.slider(
                                            "显示前多少个重要特征",
                                            min_value=min_features,
                                            max_value=max_features,
                                            value=default_features,
                                            step=1,
                                            key="regression_top_n_features"
                                        )
                                        st.dataframe(
                                            result["feature_importance_df"].head(top_n_features),
                                            use_container_width=True
                                        )
                                        st.pyplot(
                                            plot_feature_importance(result["feature_importance_df"], top_n_features))
                                    else:
                                        st.info(result["importance_message"])

                                    model_bytes = serialize_model_to_bytes(result["pipeline"])
                                    st.download_button(
                                        label="下载回归模型文件 (.pkl)",
                                        data=model_bytes,
                                        file_name="regression_model.pkl",
                                        mime="application/octet-stream",
                                        key="download_regression_model"
                                    )

                # ======================
                # 聚类分析
                # ======================
                elif page == "聚类分析":
                    st.subheader("机器学习：KMeans 聚类分析")

                    if df_cleaned.empty:
                        st.warning("请先选择至少一列数据。")
                    else:
                        numeric_columns_for_cluster = df_cleaned.select_dtypes(include=["number"]).columns.tolist()

                        if len(numeric_columns_for_cluster) >= 1:
                            cluster_features = st.multiselect(
                                "请选择用于聚类的数值特征列",
                                numeric_columns_for_cluster,
                                default=numeric_columns_for_cluster[:2] if len(
                                    numeric_columns_for_cluster) >= 2 else numeric_columns_for_cluster,
                                key="cluster_features_select"
                            )

                            if cluster_features:
                                st.markdown("### 肘部法则分析")

                                max_k_for_elbow = min(10, len(df_cleaned))

                                if max_k_for_elbow >= 2:
                                    if st.button("计算肘部法则", key="elbow_btn"):
                                        with st.spinner("正在计算不同 K 的 Inertia..."):
                                            elbow_result, elbow_error = calculate_elbow_method(
                                                df_cleaned,
                                                feature_columns=cluster_features,
                                                k_range=range(2, max_k_for_elbow + 1),
                                                random_state=42
                                            )

                                        if elbow_error:
                                            st.error(elbow_error)
                                        else:
                                            st.success("肘部法则计算完成！")
                                            st.dataframe(elbow_result["elbow_df"], use_container_width=True)

                                            fig, ax = plt.subplots(figsize=(8, 5))
                                            ax.plot(
                                                elbow_result["k_values"],
                                                elbow_result["inertias"],
                                                marker="o"
                                            )
                                            ax.set_xlabel("K")
                                            ax.set_ylabel("Inertia")
                                            ax.set_title("KMeans 肘部法则图")
                                            ax.grid(True)
                                            st.pyplot(fig)

                                max_k_for_cluster = min(10, len(df_cleaned))
                                if max_k_for_cluster >= 2:
                                    default_k = 3 if max_k_for_cluster >= 3 else 2

                                    n_clusters = st.slider(
                                        "选择聚类数 K",
                                        min_value=2,
                                        max_value=max_k_for_cluster,
                                        value=default_k,
                                        step=1,
                                        key="kmeans_cluster_slider"
                                    )

                                    if st.button("开始 KMeans 聚类", key="run_kmeans_btn"):
                                        with st.spinner("KMeans 聚类中，请稍候..."):
                                            result, cluster_error = run_kmeans_clustering(
                                                df_cleaned,
                                                feature_columns=cluster_features,
                                                n_clusters=n_clusters,
                                                random_state=42
                                            )

                                        if cluster_error:
                                            st.error(cluster_error)
                                        else:
                                            st.success("KMeans 聚类完成！")

                                            col1, col2 = st.columns(2)
                                            col1.metric("聚类数", result["n_clusters"])
                                            col2.metric("Inertia", f"{result['inertia']:.4f}")

                                            if result["silhouette_score"] is not None:
                                                st.metric("Silhouette Score", f"{result['silhouette_score']:.4f}")

                                            st.markdown("### 聚类结果预览")
                                            st.dataframe(result["result_df"].head(), use_container_width=True)

                                            cluster_csv = result["result_df"].to_csv(index=False).encode("utf-8-sig")
                                            st.download_button(
                                                label="下载聚类结果 CSV",
                                                data=cluster_csv,
                                                file_name="kmeans_result.csv",
                                                mime="text/csv",
                                                key="download_kmeans_csv"
                                            )

                                            st.markdown("### 各簇样本数量")
                                            cluster_counts = result["result_df"]["Cluster"].value_counts().sort_index()
                                            st.dataframe(cluster_counts.rename("数量"), use_container_width=True)

                                            if len(cluster_features) == 2:
                                                st.markdown("### 二维聚类散点图")
                                                fig, ax = plt.subplots(figsize=(8, 6))
                                                sns.scatterplot(
                                                    data=result["result_df"],
                                                    x=cluster_features[0],
                                                    y=cluster_features[1],
                                                    hue="Cluster",
                                                    palette="tab10",
                                                    ax=ax
                                                )
                                                ax.set_title("KMeans 聚类结果散点图")
                                                st.pyplot(fig)

                                            if len(cluster_features) >= 2:
                                                st.markdown("### PCA 二维聚类可视化")
                                                pca_fig = plot_pca_clusters(result["X_scaled"], result["labels"])
                                                if pca_fig:
                                                    st.pyplot(pca_fig)
                        else:
                            st.info("当前没有可用于 KMeans 聚类的数值列。")


# ====================== 程序入口 ======================
if __name__ == "__main__":
    main()

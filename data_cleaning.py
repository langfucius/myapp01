import pandas as pd
import numpy as np


def convert_column_types(df, type_config=None):
    """
    type_config 示例：
    {
        "age": "numeric",
        "city": "category",
        "date": "datetime",
        "name": "string"
    }
    """
    df_converted = df.copy()

    if not type_config:
        return df_converted

    for col, target_type in type_config.items():
        if col not in df_converted.columns:
            continue

        try:
            if target_type == "numeric":
                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
            elif target_type == "category":
                df_converted[col] = df_converted[col].astype("category")
            elif target_type == "datetime":
                df_converted[col] = pd.to_datetime(df_converted[col], errors="coerce")
            elif target_type == "string":
                df_converted[col] = df_converted[col].astype(str)
        except Exception:
            continue

    return df_converted


def handle_numeric_missing_values(df, method="不处理"):
    df_result = df.copy()
    numeric_columns = df_result.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        return df_result

    if method == "删除缺失行":
        df_result = df_result.dropna(subset=numeric_columns)

    elif method == "均值填充":
        for col in numeric_columns:
            df_result[col] = df_result[col].fillna(df_result[col].mean())

    elif method == "中位数填充":
        for col in numeric_columns:
            df_result[col] = df_result[col].fillna(df_result[col].median())

    elif method == "0填充":
        for col in numeric_columns:
            df_result[col] = df_result[col].fillna(0)

    elif method == "线性插值":
        df_result[numeric_columns] = df_result[numeric_columns].interpolate(
            method="linear",
            limit_direction="both"
        )

    elif method == "前向填充":
        df_result[numeric_columns] = df_result[numeric_columns].ffill()

    elif method == "后向填充":
        df_result[numeric_columns] = df_result[numeric_columns].bfill()

    return df_result


def handle_categorical_missing_values(df, method="不处理"):
    df_result = df.copy()
    categorical_columns = df_result.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not categorical_columns:
        return df_result

    if method == "删除缺失行":
        df_result = df_result.dropna(subset=categorical_columns)

    elif method == "众数填充":
        for col in categorical_columns:
            mode_value = df_result[col].mode()
            if not mode_value.empty:
                df_result[col] = df_result[col].fillna(mode_value.iloc[0])

    elif method == "Unknown填充":
        for col in categorical_columns:
            df_result[col] = df_result[col].fillna("Unknown")

    return df_result


def remove_outliers_iqr(df, multiplier=1.5):
    df_result = df.copy()
    numeric_columns = df_result.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        return df_result

    mask = pd.Series([True] * len(df_result), index=df_result.index)

    for col in numeric_columns:
        q1 = df_result[col].quantile(0.25)
        q3 = df_result[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        col_mask = df_result[col].isna() | (
            (df_result[col] >= lower_bound) & (df_result[col] <= upper_bound)
        )
        mask = mask & col_mask

    df_result = df_result[mask]
    return df_result


def remove_outliers_zscore(df, threshold=3.0):
    df_result = df.copy()
    numeric_columns = df_result.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        return df_result

    mask = pd.Series([True] * len(df_result), index=df_result.index)

    for col in numeric_columns:
        std = df_result[col].std()
        mean = df_result[col].mean()

        if pd.isna(std) or std == 0:
            continue

        z_scores = (df_result[col] - mean) / std
        col_mask = df_result[col].isna() | (z_scores.abs() <= threshold)
        mask = mask & col_mask

    df_result = df_result[mask]
    return df_result


def clean_data(
    df,
    selected_columns,
    drop_duplicates=False,
    numeric_missing_method="不处理",
    categorical_missing_method="不处理",
    outlier_method="不处理",
    type_config=None
):
    if not selected_columns:
        return pd.DataFrame()

    df_cleaned = df[selected_columns].copy()

    # 类型转换
    df_cleaned = convert_column_types(df_cleaned, type_config=type_config)

    # 缺失值处理
    df_cleaned = handle_numeric_missing_values(df_cleaned, numeric_missing_method)
    df_cleaned = handle_categorical_missing_values(df_cleaned, categorical_missing_method)

    # 异常值处理
    if outlier_method == "IQR剔除":
        df_cleaned = remove_outliers_iqr(df_cleaned)
    elif outlier_method == "Z-score剔除":
        df_cleaned = remove_outliers_zscore(df_cleaned)

    # 删除重复值
    if drop_duplicates:
        df_cleaned = df_cleaned.drop_duplicates()

    return df_cleaned


def get_missing_value_summary(df):
    if df.empty:
        return pd.DataFrame(columns=["列名", "缺失值数量", "缺失比例(%)"])

    summary = pd.DataFrame({
        "列名": df.columns,
        "缺失值数量": df.isnull().sum().values,
        "缺失比例(%)": (df.isnull().sum().values / len(df) * 100).round(2)
    })
    return summary


def get_cleaning_summary(original_df, cleaned_df):
    summary = {
        "原始行数": original_df.shape[0],
        "原始列数": original_df.shape[1],
        "清洗后行数": cleaned_df.shape[0],
        "清洗后列数": cleaned_df.shape[1],
        "减少行数": original_df.shape[0] - cleaned_df.shape[0],
        "减少列数": original_df.shape[1] - cleaned_df.shape[1],
        "原始缺失值总数": int(original_df.isnull().sum().sum()),
        "清洗后缺失值总数": int(cleaned_df.isnull().sum().sum()),
        "原始重复行数": int(original_df.duplicated().sum()),
        "清洗后重复行数": int(cleaned_df.duplicated().sum())
    }
    return summary


def get_dtype_summary(df):
    if df.empty:
        return pd.DataFrame(columns=["列名", "数据类型"])

    dtype_df = pd.DataFrame({
        "列名": df.columns,
        "数据类型": df.dtypes.astype(str).values
    })
    return dtype_df

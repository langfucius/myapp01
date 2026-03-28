import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans



# ======================
# 通用预处理
# ======================
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


# ======================
# 特征重要性
# ======================
def get_feature_importance(trained_pipeline, model_name):
    try:
        preprocessor = trained_pipeline.named_steps["preprocessor"]
        model = trained_pipeline.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()

        if model_name in ["随机森林", "决策树", "随机森林回归", "决策树回归"]:
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "特征": feature_names,
                "重要性": importances
            }).sort_values(by="重要性", ascending=False)

            return importance_df, None

        elif model_name in ["逻辑回归", "线性回归"]:
            if hasattr(model, "coef_"):
                coef = model.coef_

                if np.ndim(coef) == 1:
                    importance_values = np.abs(coef)
                else:
                    importance_values = np.mean(np.abs(coef), axis=0)

                importance_df = pd.DataFrame({
                    "特征": feature_names,
                    "重要性": importance_values
                }).sort_values(by="重要性", ascending=False)

                return importance_df, None

        elif model_name in ["KNN", "KNN回归"]:
            return None, "KNN 模型没有内置的特征重要性。"

        return None, "当前模型暂不支持特征重要性分析。"

    except Exception as e:
        return None, f"特征重要性提取失败：{str(e)}"


# ======================
# 分类模型
# ======================
def train_classification_model(
    df,
    target_column,
    model_name="随机森林",
    test_size=0.2,
    random_state=42,
    use_cv=False,
    cv_folds=5,
    use_tuning=False
):
    try:
        if target_column not in df.columns:
            return None, "目标列不存在。"

        if df.shape[0] < 10:
            return None, "样本数量太少，无法训练分类模型。"

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        if X.shape[1] == 0:
            return None, "特征列为空，请至少保留一个特征列。"

        valid_idx = y.notna()
        X = X.loc[valid_idx].copy()
        y = y.loc[valid_idx].copy()

        if len(y) < 10:
            return None, "目标列有效样本太少，无法训练分类模型。"

        if y.nunique() < 2:
            return None, "目标列类别数少于 2，无法进行分类。"

        preprocessor, numeric_features, categorical_features = build_preprocessor(X)

        if model_name == "随机森林":
            model = RandomForestClassifier(random_state=random_state)
        elif model_name == "逻辑回归":
            model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_name == "决策树":
            model = DecisionTreeClassifier(random_state=random_state)
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        else:
            return None, "不支持的模型名称。"

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        stratify_y = y_encoded if len(np.unique(y_encoded)) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y
        )

        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        cv_scores = None
        cv_mean = None
        cv_std = None
        best_params = None
        best_cv_score = None

        if use_tuning:
            param_grid = get_classification_param_grid(model_name)

            if param_grid:
                grid_search = GridSearchCV(
                    clf,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring="accuracy",
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                clf = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_cv_score = grid_search.best_score_
            else:
                clf.fit(X_train, y_train)

        else:
            clf.fit(X_train, y_train)

        if use_cv:
            cv_scores = cross_val_score(
                clf,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1
            )
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_.astype(str),
            zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        results_df = pd.DataFrame({
            "真实值": label_encoder.inverse_transform(y_test),
            "预测值": label_encoder.inverse_transform(y_pred)
        })

        feature_importance_df, importance_message = get_feature_importance(clf, model_name)

        roc_auc = None
        y_score = None
        roc_available = False

        if len(label_encoder.classes_) == 2 and hasattr(clf.named_steps["model"], "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
            y_score = y_proba
            roc_auc = roc_auc_score(y_test, y_proba)
            roc_available = True

        result = {
            "task_type": "classification",
            "model_name": model_name,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "results_df": results_df,
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "label_classes": label_encoder.classes_,
            "pipeline": clf,
            "feature_importance_df": feature_importance_df,
            "importance_message": importance_message,
            "roc_available": roc_available,
            "roc_auc": roc_auc,
            "y_test_encoded": y_test,
            "y_score": y_score,
            "target_column": target_column,
            "feature_columns": X.columns.tolist(),
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "label_encoder": label_encoder
        }

        return result, None

    except Exception as e:
        return None, str(e)



# ======================
# 回归模型
# ======================
def train_regression_model(
    df,
    target_column,
    model_name="随机森林回归",
    test_size=0.2,
    random_state=42,
    use_cv=False,
    cv_folds=5,
    use_tuning=False
):
    try:
        if target_column not in df.columns:
            return None, "目标列不存在。"

        if df.shape[0] < 10:
            return None, "样本数量太少，无法训练回归模型。"

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        if X.shape[1] == 0:
            return None, "特征列为空，请至少保留一个特征列。"

        valid_idx = y.notna()
        X = X.loc[valid_idx].copy()
        y = y.loc[valid_idx].copy()

        if len(y) < 10:
            return None, "目标列有效样本太少，无法训练回归模型。"

        if not pd.api.types.is_numeric_dtype(y):
            return None, "回归任务的目标列必须是数值型。"

        preprocessor, numeric_features, categorical_features = build_preprocessor(X)

        if model_name == "线性回归":
            model = LinearRegression()
        elif model_name == "随机森林回归":
            model = RandomForestRegressor(random_state=random_state)
        elif model_name == "决策树回归":
            model = DecisionTreeRegressor(random_state=random_state)
        elif model_name == "KNN回归":
            model = KNeighborsRegressor()
        else:
            return None, "不支持的回归模型名称。"

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        reg = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        cv_scores = None
        cv_mean = None
        cv_std = None
        best_params = None
        best_cv_score = None

        if use_tuning:
            param_grid = get_regression_param_grid(model_name)

            if param_grid:
                grid_search = GridSearchCV(
                    reg,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring="r2",
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                reg = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_cv_score = grid_search.best_score_
            else:
                reg.fit(X_train, y_train)

        else:
            reg.fit(X_train, y_train)

        if use_cv:
            cv_scores = cross_val_score(
                reg,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="r2",
                n_jobs=-1
            )
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results_df = pd.DataFrame({
            "真实值": y_test.values,
            "预测值": y_pred,
            "残差": y_test.values - y_pred
        })

        feature_importance_df, importance_message = get_feature_importance(reg, model_name)

        result = {
            "task_type": "regression",
            "model_name": model_name,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "results_df": results_df,
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "pipeline": reg,
            "feature_importance_df": feature_importance_df,
            "importance_message": importance_message,
            "y_test": y_test.values,
            "y_pred": y_pred,
            "target_column": target_column,
            "feature_columns": X.columns.tolist(),
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "best_params": best_params,
            "best_cv_score": best_cv_score
        }

        return result, None

    except Exception as e:
        return None, str(e)


# ======================
# KMeans 聚类
# ======================
def run_kmeans_clustering(
    df,
    feature_columns,
    n_clusters=3,
    random_state=42
):
    try:
        if not feature_columns:
            return None, "请至少选择一个特征列用于聚类。"

        cluster_df = df[feature_columns].copy()

        numeric_columns = cluster_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_columns) != len(feature_columns):
            return None, "KMeans 仅支持数值列，请重新选择特征列。"

        if cluster_df.isnull().sum().sum() > 0:
            cluster_df = cluster_df.fillna(cluster_df.median(numeric_only=True))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)

        if len(cluster_df) <= n_clusters:
            return None, "样本数量必须大于聚类数。"

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)

        result_df = cluster_df.copy()
        result_df["Cluster"] = labels

        sil_score = silhouette_score(X_scaled, labels) if n_clusters >= 2 else None

        result = {
            "n_clusters": n_clusters,
            "inertia": model.inertia_,
            "silhouette_score": sil_score,
            "result_df": result_df,
            "cluster_centers": model.cluster_centers_,
            "labels": labels,
            "X_scaled": X_scaled
        }

        return result, None

    except Exception as e:
        return None, str(e)


# ======================
# 肘部法则
# ======================
def calculate_elbow_method(
    df,
    feature_columns,
    k_range=range(2, 11),
    random_state=42
):
    try:
        if not feature_columns:
            return None, "请至少选择一个特征列。"

        elbow_df = df[feature_columns].copy()

        numeric_columns = elbow_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_columns) != len(feature_columns):
            return None, "肘部法则分析仅支持数值列。"

        if elbow_df.isnull().sum().sum() > 0:
            elbow_df = elbow_df.fillna(elbow_df.median(numeric_only=True))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(elbow_df)

        k_values = []
        inertias = []

        for k in k_range:
            if k < len(elbow_df):
                model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                model.fit(X_scaled)
                k_values.append(k)
                inertias.append(model.inertia_)

        result = {
            "k_values": k_values,
            "inertias": inertias,
            "elbow_df": pd.DataFrame({
                "K": k_values,
                "Inertia": inertias
            })
        }

        return result, None

    except Exception as e:
        return None, str(e)

def get_classification_param_grid(model_name):
    if model_name == "随机森林":
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5]
        }
    elif model_name == "逻辑回归":
        return {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs"],
            "model__max_iter": [1000]
        }
    elif model_name == "决策树":
        return {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    elif model_name == "KNN":
        return {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"]
        }
    return {}


def get_regression_param_grid(model_name):
    if model_name == "随机森林回归":
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5]
        }
    elif model_name == "线性回归":
        return {}
    elif model_name == "决策树回归":
        return {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    elif model_name == "KNN回归":
        return {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"]
        }
    return {}


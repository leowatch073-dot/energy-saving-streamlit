import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import FEATURE_COLS

#=============获取单宿舍小时数据=================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为小时级能耗数据补充时间特征。

    目前生成的特征包括：
    - hour: 小时（0~23）
    - dow: 星期（0~6，周一为 0）
    - is_weekend: 是否周末（0/1）

    参数
    ----
    df : pd.DataFrame
        至少需要包含 timestamp 列。

    返回
    ----
    pd.DataFrame
        在原数据基础上补充时间特征后的副本。
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

#=============Ridge baseline 训练=================
def train_ridge_baseline(hourly_df: pd.DataFrame, alpha: float = 1.0):
    """
    训练 Ridge 基线预测模型。
    """
    if hourly_df is None or hourly_df.empty:
        raise ValueError("hourly_df 为空，无法训练 baseline。")

    df = add_time_features(hourly_df)

    required = {"timestamp", "energy_kwh"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"训练 baseline 缺少必需列: {miss}")

    df = df.dropna(subset=["energy_kwh"]).copy()
    if df.empty:
        raise ValueError("energy_kwh 全为空，无法训练 baseline。")

    X = df[FEATURE_COLS].copy()
    y = pd.to_numeric(df["energy_kwh"], errors="coerce")

    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    if len(X) == 0:
        raise ValueError("有效训练样本为空，无法训练 baseline。")

    numeric_features = ["hour", "dow", "is_weekend"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )

    pipe.fit(X, y)

    model = pipe.named_steps["model"]

    info = {
        "n_samples": int(len(df)),
        "feature_cols": FEATURE_COLS,
        "intercept": float(model.intercept_),
        "coef": [float(x) for x in np.ravel(model.coef_)],
        "alpha": float(alpha),
    }

    return pipe, info

#=============baseline预测=================
def predict_baseline(model, future_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用已训练好的 baseline 模型进行预测。

    参数
    ----
    model :
        train_ridge_baseline 返回的训练好模型。
    future_df : pd.DataFrame
        至少需要包含 timestamp 列。

    返回
    ----
    pd.DataFrame
        在输入数据基础上新增 baseline_pred 列。
    """
    if future_df is None or future_df.empty:
        raise ValueError("future_df 为空，无法进行 baseline 预测。")

    df = add_time_features(future_df)

    if "timestamp" not in df.columns:
        raise ValueError("future_df 缺少 timestamp 列。")

    X = df[FEATURE_COLS].copy()
    pred = model.predict(X)

    out = df.copy()
    out["baseline_pred"] = np.clip(pred, 0, None)
    return out
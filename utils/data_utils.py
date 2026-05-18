#处理原始能耗数据、时间列、小时序列
import io
import numpy as np
import pandas as pd

#==========模拟能耗数据生成=============
def generate_energy_sample(n_days=28, dorms=None, seed=42):
    
    rng = np.random.default_rng(seed)
    if dorms is None:
        dorms = ["TJ_SIPING_B3_0612", "TJ_SIPING_B3_0613", "TJ_SIPING_B3_0614"]

    idx = pd.date_range(end=pd.Timestamp.today().floor("h"), periods=n_days * 24, freq="h")
    rows = []
    base_map = {
        dorms[0]: 1.6,
        dorms[1]: 1.1,
        dorms[2]: 0.8,
    }

    for d in dorms:
        base = base_map.get(d, 1.0)
        for ts in idx:
            hour = ts.hour
            temp = 10 + 12 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 1.2)
            occ = 0.5 + 0.5 * np.sin((hour - 18) / 24 * 2 * np.pi)
            noise = rng.normal(0, 0.15)
            e = max(
                0,
                base
                + 0.06 * max(temp - 20, 0)
                + 0.35 * occ
                + (0.25 if hour in [12, 13, 14, 21, 22, 23] else 0)
                + noise,
            )
            rows.append([ts, d, round(float(e), 3), round(float(temp), 1)])

    return pd.DataFrame(rows, columns=["timestamp", "dorm_id", "energy_kwh", "outdoor_temp_c"])

#=========时间列解析==========
def parse_timestamp_hour(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.floor("h")

    ts_from_str = pd.to_datetime(series, errors="coerce")
    s_num = pd.to_numeric(series, errors="coerce")

    excel_mask = s_num.between(20000, 60000)

    ts_from_num = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if excel_mask.any():
        ts_from_num.loc[excel_mask] = (
            pd.to_datetime("1899-12-30")
            + pd.to_timedelta(s_num.loc[excel_mask], unit="D")
        )

    ts = ts_from_str.where(ts_from_str.notna(), ts_from_num)
    return ts.dt.floor("h")

#============CSV / DataFrame 导入==============
def load_energy_df(df: pd.DataFrame, align_to_hour=True, strict_hour=False):
    out = df.copy()

    required = {"timestamp", "energy_kwh"}
    miss = required - set(out.columns)
    if miss:
        raise ValueError(f"缺少必需列: {miss}")

    if "dorm_id" not in out.columns:
        out["dorm_id"] = "DORM_001"
    if "device_id" not in out.columns:
        out["device_id"] = "AC_001"

    out["timestamp"] = parse_timestamp_hour(out["timestamp"]) if align_to_hour else pd.to_datetime(
        out["timestamp"], errors="coerce"
    )

    if strict_hour:
        raw_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        bad = raw_ts.notna() & (
            (raw_ts.dt.minute != 0) | (raw_ts.dt.second != 0) | (raw_ts.dt.microsecond != 0)
        )
        if bad.any():
            n_bad = int(bad.sum())
            raise ValueError(f"检测到 {n_bad} 条记录不是整点小时；请先聚合或关闭 strict_hour。")

    out["energy_kwh"] = pd.to_numeric(out["energy_kwh"], errors="coerce")
    if "outdoor_temp_c" in out.columns:
        out["outdoor_temp_c"] = pd.to_numeric(out["outdoor_temp_c"], errors="coerce")

    out = out.dropna(subset=["timestamp", "energy_kwh", "dorm_id"])
    out["hour"] = out["timestamp"].dt.floor("h")

    agg = {"energy_kwh": "sum"}
    if "outdoor_temp_c" in out.columns:
        agg["outdoor_temp_c"] = "mean"

    hourly = (
        out.groupby(["dorm_id", "hour"], as_index=False)
        .agg(agg)
        .rename(columns={"hour": "timestamp"})
        .sort_values(["dorm_id", "timestamp"])
        .reset_index(drop=True)
    )

    stats = {
        "n_rows_raw": int(len(df)),
        "n_rows_clean": int(len(out)),
        "n_rows_hourly": int(len(hourly)),
        "n_dorms": int(hourly["dorm_id"].nunique()) if len(hourly) else 0,
        "time_min": hourly["timestamp"].min() if len(hourly) else None,
        "time_max": hourly["timestamp"].max() if len(hourly) else None,
    }
    return hourly, stats

def load_energy_csv(file_like, align_to_hour=True, strict_hour=False):
    if isinstance(file_like, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(file_like))
    else:
        df = pd.read_csv(file_like)
    return load_energy_df(df, align_to_hour=align_to_hour, strict_hour=strict_hour)
#=============小时对齐与聚合===============
def week_start(ts):
    """
    取某个时间对应周的周一 00:00。
    """
    t = pd.Timestamp(ts)
    return (t - pd.Timedelta(days=t.weekday())).normalize()


def fill_hours(df_hourly: pd.DataFrame, dorm_id: str) -> pd.DataFrame:
    d = df_hourly[df_hourly["dorm_id"] == dorm_id].copy()
    if d.empty:
        return d

    idx = pd.date_range(d["timestamp"].min(), d["timestamp"].max(), freq="h")
    d = d.set_index("timestamp").reindex(idx)
    d.index.name = "timestamp"
    d["dorm_id"] = dorm_id
    if "energy_kwh" in d.columns:
        d["energy_kwh"] = d["energy_kwh"].fillna(0)
    return d.reset_index()

#=============获取单宿舍小时数据===============
def today_str():
    """
    返回当前日期字符串 YYYY-MM-DD。
    """
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def get_dorm_hourly(hourly_df: pd.DataFrame, dorm_id: str) -> pd.DataFrame:
    """
    获取单个宿舍的小时级数据，并按时间排序。
    """
    return (
        hourly_df[hourly_df["dorm_id"] == dorm_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
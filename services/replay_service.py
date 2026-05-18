import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_CLUSTER_ARMS, DORM_CS_PROFILE
from models.scoring import classify_cluster
from models.bandit import parse_state_vector, linucb_choose_arm
from models.cluster_rules import (
    enrich_score_record,
    classify_cluster_v2_from_fixed_scores,
    classify_cluster_v2_from_pool_distribution,
)
from services.sqlite_service import save_sqlite_intervention_event


def _latest_scores_map(scores):
    out = {}
    if not scores:
        return out
    for rec in scores:
        if not isinstance(rec, dict):
            continue
        dorm = rec.get("dorm_id")
        ts = rec.get("ts", "")
        if dorm is None:
            continue
        if dorm not in out or str(ts) > str(out[dorm].get("ts", "")):
            out[dorm] = rec
    return out


def clear_scores_for_all_dorms():
    return []


def _pick_timestamp_col(df: pd.DataFrame):
    for c in ["timestamp", "timestamp_hour"]:
        if c in df.columns:
            return c
    return None


def _safe_numeric_mean(s):
    return float(pd.to_numeric(s, errors="coerce").dropna().mean()) if s is not None and len(s) else 0.0


def _compute_bs_from_hourly(hourly_sub: pd.DataFrame, tau: float = 0.30) -> float:
    """
    兼容旧主程序的简化 BS：
    用近一周平均小时电耗做一个 0~1 的压缩映射。
    """
    if hourly_sub is None or hourly_sub.empty:
        return 0.5

    df = hourly_sub.copy()
    ts_col = _pick_timestamp_col(df)
    if ts_col is None:
        return 0.5
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        return 0.5

    val_col = None
    for c in ["energy_kwh", "kwh", "actual_kwh"]:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        return 0.5

    recent = df.sort_values(ts_col).tail(24 * 7)
    avg_kwh = _safe_numeric_mean(recent[val_col])
    if math.isnan(avg_kwh):
        return 0.5

    tau = max(float(tau), 1e-6)
    bs = 1.0 - math.exp(-avg_kwh / tau)
    return max(0.0, min(1.0, float(bs)))


# 兼容两种调用：
# 1) 旧主程序：batch_generate_scores_for_all_dorms(df_all, tau=0.30) -> (n, msg)
# 2) 新 admin 页面：batch_generate_scores_for_all_dorms(dorm_ids=..., dorm_cs_profile=..., hourly_df=..., ...) -> list[dict]

def batch_generate_scores_for_all_dorms(*args, **kwargs):
    if args and isinstance(args[0], pd.DataFrame):
        # 旧接口模式
        hourly_df = args[0]
        tau = kwargs.get("tau", 0.30)
        dorm_cs_profile = kwargs.get("dorm_cs_profile", DORM_CS_PROFILE)
        now_ts = kwargs.get("now_ts", None)

        if hourly_df is None or hourly_df.empty or "dorm_id" not in hourly_df.columns:
            return 0, "hourly_df 为空或缺少 dorm_id。"

        dorm_ids = sorted(pd.Series(hourly_df["dorm_id"]).dropna().astype(str).unique().tolist())
        records = _generate_score_records(
            dorm_ids=dorm_ids,
            dorm_cs_profile=dorm_cs_profile,
            hourly_df=hourly_df,
            outcome_df=kwargs.get("outcome_df"),
            tau=tau,
            now_ts=now_ts,
        )
        st.session_state["scores"] = records
        return len(records), "已写入 session_state['scores']"

    # 新接口模式
    return _generate_score_records(
        dorm_ids=kwargs.get("dorm_ids", args[0] if len(args) > 0 else []),
        dorm_cs_profile=kwargs.get("dorm_cs_profile", DORM_CS_PROFILE),
        hourly_df=kwargs.get("hourly_df", args[1] if len(args) > 1 else None),
        outcome_df=kwargs.get("outcome_df", None),
        tau=kwargs.get("tau", 0.30),
        now_ts=kwargs.get("now_ts", None),
    )


def _generate_score_records(
    dorm_ids,
    dorm_cs_profile,
    hourly_df,
    outcome_df=None,
    tau=0.30,
    now_ts=None,
):
    records = []
    ts_now = pd.Timestamp(now_ts) if now_ts is not None else pd.Timestamp.now()
    hourly_df = hourly_df.copy() if isinstance(hourly_df, pd.DataFrame) else pd.DataFrame()

    for dorm_id in dorm_ids or []:
        cs = float((dorm_cs_profile or {}).get(dorm_id, 0.5))

        dorm_hourly = (
            hourly_df[
                hourly_df.get("dorm_id", pd.Series(dtype=str)).astype(str) == str(dorm_id)
            ].copy()
            if not hourly_df.empty and "dorm_id" in hourly_df.columns
            else pd.DataFrame()
        )

        bs = _compute_bs_from_hourly(dorm_hourly, tau=tau)

        # 先保留旧规则结果
        base_cluster_type, base_dist2 = classify_cluster(cs, bs)

        # 再做轻量修正，不推翻原逻辑
        cluster_type, dist2 = classify_cluster_v2_from_fixed_scores(
            cs=cs,
            bs=bs,
            fallback_cluster_type=base_cluster_type,
            fallback_dist2=base_dist2,
        )

        score = {
            "dorm_id": str(dorm_id),
            "ts": ts_now.strftime("%Y-%m-%d %H:%M:%S"),
            "CS": round(float(cs), 6),
            "BS_energy_week": round(float(bs), 6),
            "cluster_type": cluster_type,
            "cluster_dist2": None if dist2 is None else round(float(dist2), 6),
            "tau": float(tau),
            "score_source": "real_dorm_fixed_cs_bs",
        }

        score = enrich_score_record(
            score=score,
            cluster_type=cluster_type,
            cluster_dist2=dist2,
            reason_source="fixed_cs_bs_v2",
        )
        records.append(score)

    return records

def _generate_score_records(
    dorm_ids,
    dorm_cs_profile,
    hourly_df,
    outcome_df=None,
    tau=0.30,
    now_ts=None,
):
    records = []
    ts_now = pd.Timestamp(now_ts) if now_ts is not None else pd.Timestamp.now()
    hourly_df = hourly_df.copy() if isinstance(hourly_df, pd.DataFrame) else pd.DataFrame()

    for dorm_id in dorm_ids or []:
        cs = float((dorm_cs_profile or {}).get(dorm_id, 0.5))

        dorm_hourly = (
            hourly_df[
                hourly_df.get("dorm_id", pd.Series(dtype=str)).astype(str) == str(dorm_id)
            ].copy()
            if not hourly_df.empty and "dorm_id" in hourly_df.columns
            else pd.DataFrame()
        )

        bs = _compute_bs_from_hourly(dorm_hourly, tau=tau)

        # 先保留旧规则结果
        base_cluster_type, base_dist2 = classify_cluster(cs, bs)

        # 再做轻量修正，不推翻原逻辑
        cluster_type, dist2 = classify_cluster_v2_from_fixed_scores(
            cs=cs,
            bs=bs,
            fallback_cluster_type=base_cluster_type,
            fallback_dist2=base_dist2,
        )

        score = {
            "dorm_id": str(dorm_id),
            "ts": ts_now.strftime("%Y-%m-%d %H:%M:%S"),
            "CS": round(float(cs), 6),
            "BS_energy_week": round(float(bs), 6),
            "cluster_type": cluster_type,
            "cluster_dist2": None if dist2 is None else round(float(dist2), 6),
            "tau": float(tau),
            "score_source": "real_dorm_fixed_cs_bs",
        }

        score = enrich_score_record(
            score=score,
            cluster_type=cluster_type,
            cluster_dist2=dist2,
            reason_source="fixed_cs_bs_v2",
        )
        records.append(score)

    return records

def _week_start_for_dt(dt):
    t = pd.Timestamp(dt)
    return (t - pd.Timedelta(days=t.weekday())).normalize()


def _is_weekend(dt):
    t = pd.Timestamp(dt)
    return int(t.dayofweek >= 5)


def _recent_diff_3h(dorm_out: pd.DataFrame, decision_time) -> float:
    if dorm_out is None or dorm_out.empty:
        return 0.0
    df = dorm_out.copy()
    ts_col = _pick_timestamp_col(df)
    if ts_col is None:
        return 0.0

    actual_col = None
    for c in ["energy_kwh", "kwh", "actual_kwh"]:
        if c in df.columns:
            actual_col = c
            break
    base_col = None
    for c in ["baseline_pred", "baseline_kwh", "baseline"]:
        if c in df.columns:
            base_col = c
            break
    if actual_col is None or base_col is None:
        return 0.0

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    t0 = pd.Timestamp(decision_time)
    t1 = t0 - pd.Timedelta(hours=3)
    sub = df[(df[ts_col] < t0) & (df[ts_col] >= t1)].copy()
    if sub.empty:
        return 0.0
    actual = pd.to_numeric(sub[actual_col], errors="coerce").fillna(0.0)
    base = pd.to_numeric(sub[base_col], errors="coerce").fillna(0.0)
    return float((actual - base).mean())


def _cluster_type_from_scores_or_compute(latest_score, cs=None, bs=None):
    if isinstance(latest_score, dict):
        c = latest_score.get("cluster_type")
        if c:
            return c

        cs0 = latest_score.get("CS", cs)
        bs0 = latest_score.get("BS_energy_week", bs)

        if cs0 is not None and bs0 is not None:
            base_c, base_d = classify_cluster(cs0, bs0)
            c, _ = classify_cluster_v2_from_fixed_scores(
                cs=cs0,
                bs=bs0,
                fallback_cluster_type=base_c,
                fallback_dist2=base_d,
            )
            return c

    if cs is not None and bs is not None:
        base_c, base_d = classify_cluster(cs, bs)
        c, _ = classify_cluster_v2_from_fixed_scores(
            cs=cs,
            bs=bs,
            fallback_cluster_type=base_c,
            fallback_dist2=base_d,
        )
        return c

    return None


def _candidates_for_cluster(cluster_type, cluster_arms_config=None):
    if cluster_arms_config and cluster_type in cluster_arms_config:
        arms = cluster_arms_config.get(cluster_type, [])
        return list(arms) if arms else []
    return list(DEFAULT_CLUSTER_ARMS.get(cluster_type, []))


import pandas as pd
import numpy as np


def build_pool_thresholds(user_pool_df):
    """
    根据用户池自身分布，动态计算 AS / BS 的高低阈值。
    这样不会被固定 0.5 阈值限制，更适合真实统计数据。
    """
    df = user_pool_df.copy()

    as_series = pd.to_numeric(df["AS"], errors="coerce").dropna()
    bs_series = pd.to_numeric(df["BS"], errors="coerce").dropna()

    if len(as_series) == 0 or len(bs_series) == 0:
        raise ValueError("用户池中的 AS/BS 无法计算阈值。")

    thresholds = {
        "AS_p33": float(as_series.quantile(0.33)),
        "AS_p50": float(as_series.quantile(0.50)),
        "AS_p67": float(as_series.quantile(0.67)),
        "BS_p33": float(bs_series.quantile(0.33)),
        "BS_p50": float(bs_series.quantile(0.50)),
        "BS_p67": float(bs_series.quantile(0.67)),
    }
    return thresholds


def classify_cluster_from_pool_distribution(as_val, bs_val, thresholds):
    """
    基于用户池分布进行分群（v2）。
    返回值必须是二元组: (cluster_type, dist2)
    """
    ret = classify_cluster_v2_from_pool_distribution(
        as_val=as_val,
        bs_val=bs_val,
        thresholds=thresholds,
    )
    # 防呆：确保始终返回两个值
    if ret is None:
        return "E", None

    if isinstance(ret, tuple):
        if len(ret) >= 2:
            return ret[0], ret[1]
        if len(ret) == 1:
            return ret[0], None

    # 万一返回的是单个字符串，例如 "A"
    return str(ret), None

def load_user_pool_from_excel(excel_path, sheet_name=0, id_prefix="POOL"):
    """
    从 Excel 读取用户池，并标准化为系统可用字段。

    输入要求：
    - 至少包含 AS 和 BS 两列

    输出字段：
    - dorm_id
    - AS
    - BS
    - CS
    - BS_energy_week
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name).copy()

    # 清理列名空格
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["AS", "BS"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"用户池文件缺少必要列：{missing}，当前列为：{list(df.columns)}")

    # 删除 AS/BS 缺失的记录
    df = df.dropna(subset=["AS", "BS"]).copy()

    # 数值化从用户池生成
    df["AS"] = pd.to_numeric(df["AS"], errors="coerce")
    df["BS"] = pd.to_numeric(df["BS"], errors="coerce")
    df = df.dropna(subset=["AS", "BS"]).copy()

    # 限制到 [0, 1]，避免异常值影响
    df["AS"] = df["AS"].clip(0, 1)
    df["BS"] = df["BS"].clip(0, 1)

    # 自动生成虚拟 dorm_id
    df["dorm_id"] = [f"{id_prefix}_{i+1:04d}" for i in range(len(df))]

    # 映射成系统当前 score 所需字段
    df["CS"] = df["AS"]
    df["BS_energy_week"] = df["BS"]

    return df.reset_index(drop=True)

def batch_generate_scores_from_user_pool(user_pool_df, tau=0.30, now_ts=None):
    """
    从用户池 DataFrame 批量生成 scores。
    输出格式与当前 replay / admin 页兼容。

    新版逻辑：
    - 不再使用固定 0.5 阈值
    - 改为使用用户池自身分布（33% / 67% 分位）进行分群
    """
    if user_pool_df is None or len(user_pool_df) == 0:
        return [], "用户池为空，未生成任何 scores。"

    df = user_pool_df.copy()

    needed = ["dorm_id", "CS", "BS_energy_week", "AS", "BS"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return [], f"用户池缺少必要字段：{missing}"

    if now_ts is None:
        now_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # 先根据整份用户池计算动态阈值
    thresholds = build_pool_thresholds(df)

    scores = []

    for _, row in df.iterrows():
        dorm_id = str(row["dorm_id"])
        cs = float(row["CS"])
        bs = float(row["BS_energy_week"])

        cluster_ret = classify_cluster_from_pool_distribution(
            as_val=row["AS"],
            bs_val=row["BS"],
            thresholds=thresholds,
        )

        if cluster_ret is None:
            cluster_type, dist2 = "E", None
        elif isinstance(cluster_ret, tuple):
            if len(cluster_ret) >= 2:
                cluster_type, dist2 = cluster_ret[0], cluster_ret[1]
            elif len(cluster_ret) == 1:
                cluster_type, dist2 = cluster_ret[0], None
            else:
                cluster_type, dist2 = "E", None
        else:
            cluster_type, dist2 = str(cluster_ret), None

        score = {
            "dorm_id": dorm_id,
            "ts": now_ts,
            "CS": round(cs, 6),
            "BS_energy_week": round(bs, 6),
            "cluster_type": cluster_type,
            "cluster_dist2": None if dist2 is None else round(float(dist2), 6),
            "tau": float(tau),
            "AS_raw": round(float(row["AS"]), 6),
            "BS_raw": round(float(row["BS"]), 6),
            "score_source": "user_pool_excel",
        }

        score = enrich_score_record(
            score=score,
            cluster_type=cluster_type,
            cluster_dist2=dist2,
            reason_source="pool_distribution_v2",
        )
        scores.append(score)
    return scores, f"已从用户池生成 {len(scores)} 条 scores。"


def batch_generate_scores_for_real_dorms_from_user_pool(
    real_dorm_ids,
    user_pool_df,
    tau=0.30,
    now_ts=None,
    random_state=42,
):
    """
    从用户池中抽样画像，并映射到真实 dorm_id，生成可供 replay 直接使用的 scores。

    作用：
    1. replay 当前仍按 outcome_logs_df 中的真实 dorm_id 工作
    2. 因此不能直接使用 POOL_0001 这类虚拟 dorm_id
    3. 本函数会把用户池中的 AS/BS 画像赋给真实 dorm，再生成 scores

    参数
    ----
    real_dorm_ids : list[str]
        outcome_logs_df 中真实存在的 dorm_id 列表
    user_pool_df : pd.DataFrame
        由 load_user_pool_from_excel() 读取的用户池数据
    tau : float
        分群模糊区参数
    now_ts : str or None
        score 时间戳；为空则自动取当前时间
    random_state : int
        抽样随机种子

    返回
    ----
    scores : list[dict]
    msg : str
    """
    if real_dorm_ids is None or len(real_dorm_ids) == 0:
        return [], "真实 dorm_id 为空，无法生成 scores。"

    if user_pool_df is None or len(user_pool_df) == 0:
        return [], "用户池为空，无法生成 scores。"

    df = user_pool_df.copy()
    real_dorm_ids = [str(x) for x in real_dorm_ids]

    n = len(real_dorm_ids)

    # 若用户池记录不足，则允许放回抽样；否则无放回抽样
    if len(df) < n:
        sampled = df.sample(n=n, replace=True, random_state=int(random_state)).reset_index(drop=True)
    else:
        sampled = df.sample(n=n, replace=False, random_state=int(random_state)).reset_index(drop=True)

    # 将抽样得到的画像映射到真实 dorm_id
    sampled = sampled.copy()
    sampled["dorm_id"] = real_dorm_ids

    # 复用现有的用户池 score 生成逻辑
    scores, msg = batch_generate_scores_from_user_pool(
        user_pool_df=sampled,
        tau=tau,
        now_ts=now_ts,
    )

    msg = f"已将用户池画像映射到 {len(real_dorm_ids)} 个真实 dorm，并生成 {len(scores)} 条 scores。"
    return scores, msg

def generate_simulated_user_pool_from_base_samples(
    base_user_pool_df,
    target_n=1000,
    noise_scale=0.03,
    random_state=42,
    id_prefix="SIM",
):
    """
    基于原始用户池样本，采用“有放回重采样 + 小扰动”的方式，
    生成更大的模拟用户池。
    """
    if base_user_pool_df is None or len(base_user_pool_df) == 0:
        raise ValueError("base_user_pool_df 为空，无法生成模拟用户池。")

    df = base_user_pool_df.copy()

    required_cols = ["AS", "BS"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"基础用户池缺少必要列：{missing}")

    rng = np.random.default_rng(int(random_state))

    sampled = df.sample(n=int(target_n), replace=True, random_state=int(random_state)).reset_index(drop=True)

    sampled["AS"] = pd.to_numeric(sampled["AS"], errors="coerce").fillna(0.5)
    sampled["BS"] = pd.to_numeric(sampled["BS"], errors="coerce").fillna(0.5)

    # 小扰动
    sampled["AS"] = sampled["AS"] + rng.normal(0, float(noise_scale), size=len(sampled))
    sampled["BS"] = sampled["BS"] + rng.normal(0, float(noise_scale), size=len(sampled))

    # 截断到 [0, 1]
    sampled["AS"] = sampled["AS"].clip(0, 1)
    sampled["BS"] = sampled["BS"].clip(0, 1)

    sampled["dorm_id"] = [f"{id_prefix}_{i+1:05d}" for i in range(len(sampled))]
    sampled["CS"] = sampled["AS"]
    sampled["BS_energy_week"] = sampled["BS"]
    sampled["pool_source"] = "simulated_from_base_pool"
    print("DEBUG function exists: generate_simulated_user_pool_from_base_samples")
   
    return sampled.reset_index(drop=True)
    

def batch_generate_scores_from_sim_user_pool(sim_user_pool_df, tau=0.30, now_ts=None):
    """
    从更大的模拟用户池直接生成 scores。
    本质上复用 batch_generate_scores_from_user_pool。
    """
    return batch_generate_scores_from_user_pool(
        user_pool_df=sim_user_pool_df,
        tau=tau,
        now_ts=now_ts,
    )

def batch_generate_interventions_replay(
    outcome_logs_df,
    scores,
    linucb_params,
    start_date,
    end_date,
    send_hour,
    cluster_arms_config=None,
    alpha_ucb=1.0,
    write_sqlite=True,
):
    logs = []
    if outcome_logs_df is None or outcome_logs_df.empty:
        return 0, "outcome_logs_df 为空，无法生成。"

    df = outcome_logs_df.copy()
    ts_col = _pick_timestamp_col(df)
    if "dorm_id" not in df.columns or ts_col is None:
        return 0, "outcome_logs 缺少 dorm_id / timestamp 列。"

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        return 0, "有效 timestamp 数据为空。"
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})

    outcome_by_dorm = {
        str(dorm_id): sub.sort_values("timestamp").reset_index(drop=True)
        for dorm_id, sub in df.groupby("dorm_id")
    }
    latest_scores_map = _latest_scores_map(scores)

    start_dt = pd.to_datetime(start_date, errors="coerce")
    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_dt) or pd.isna(end_dt) or end_dt < start_dt:
        return 0, "start_date / end_date 非法。"
    day_list = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="D")

    if linucb_params is not None:
        try:
            linucb_params["alpha"] = float(alpha_ucb)
        except Exception:
            pass

    sqlite_n = 0
    for day in day_list:
        decision_time = pd.Timestamp(day) + pd.Timedelta(hours=int(send_hour))
        for dorm_id, dorm_out in outcome_by_dorm.items():
            latest_score = latest_scores_map.get(dorm_id, {})
            cs = float(latest_score.get("CS", 0.5))
            bs = float(latest_score.get("BS_energy_week", 0.5))
            cluster_type = _cluster_type_from_scores_or_compute(latest_score, cs, bs)
            if cluster_type is None:
                continue
            candidate_arms = _candidates_for_cluster(cluster_type, cluster_arms_config)
            if not candidate_arms:
                continue
            recent_diff = _recent_diff_3h(dorm_out, decision_time)
            is_weekend = _is_weekend(decision_time)
            x = parse_state_vector(cs, bs, recent_diff, is_weekend)
            best_arm, best_detail, details = linucb_choose_arm(linucb_params, candidate_arms, x)
            row = {
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dorm_id": dorm_id,
                "cluster_type": cluster_type,
                "arm_id": best_arm,
                "algo_type": "LinUCB",
                "policy_version": f"alpha={float(alpha_ucb):.2f}",
                "ucb_score": float(best_detail["score"]),
                "p_hat": float(best_detail["mu"]),
                "bonus": float(best_detail["bonus"]),
                "reward_sum_12h": None,
                "update_timestamp": None,
                "decision_t0_hour": decision_time.strftime("%Y-%m-%d %H:%M:%S"),
                "state_json": pd.Series(
                    {
                        "CS": float(cs),
                        "BS_energy_week": float(bs),
                        "recent_diff_3h": float(recent_diff),
                        "is_weekend": int(is_weekend),
                    }
                ).to_json(force_ascii=False),
            }
            logs.append(row)
            if write_sqlite:
                save_sqlite_intervention_event(
                    plug_id=dorm_id,
                    decision_time=row["decision_t0_hour"],
                    current_consumption=float(cs),
                    baseline_consumption=float(bs),
                    recent_diff_3h=float(recent_diff),
                    is_weekend=int(is_weekend),
                    time_block=f"replay_h{int(send_hour)}",
                    eligible_templates=candidate_arms,
                    assigned_template=best_arm,
                    assignment_mode="LinUCB",
                    window_hours=12,
                    reward_value=None,
                    reward_ready_flag=0,
                )
                sqlite_n += 1

    existing = st.session_state.get("intervention_logs", [])
    if existing is None:
        existing = []
    st.session_state["intervention_logs"] = list(existing) + logs

    n = len(logs)
    if n == 0:
        return 0, "未生成任何干预事件，请检查日期范围、scores、cluster 配置或 outcome 数据。"
    return n, f"ok | sqlite={sqlite_n}"


def is_missing_value(x) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return x is None


def reward_sum_for_intervention(outcome_logs_df: pd.DataFrame, dorm_id: str, t0, hours=12) -> float:
    if outcome_logs_df is None or outcome_logs_df.empty:
        return 0.0
    df = outcome_logs_df.copy()
    ts_col = _pick_timestamp_col(df)
    if ts_col is None or "dorm_id" not in df.columns or "reward_value" not in df.columns:
        return 0.0
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    start = pd.Timestamp(t0)
    end = start + pd.Timedelta(hours=int(hours))
    sub = df[(df["dorm_id"].astype(str) == str(dorm_id)) & (df[ts_col] >= start) & (df[ts_col] < end)].copy()
    if sub.empty:
        return 0.0
    vals = pd.to_numeric(sub["reward_value"], errors="coerce").fillna(0.0)
    return float(vals.sum())


def get_decision_t0(dorm_out: pd.DataFrame, demo_update=True):
    now = pd.Timestamp.now()
    if dorm_out is None or dorm_out.empty:
        return now
    df = dorm_out.copy()
    ts_col = _pick_timestamp_col(df)
    if ts_col is None:
        return now
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    if df.empty:
        return now
    t_last = df[ts_col].max()
    if demo_update:
        return t_last - pd.Timedelta(hours=12)
    return t_last


def get_auto_replay_range(hourly_df: pd.DataFrame):
    if hourly_df is None or hourly_df.empty:
        return None, None
    df = hourly_df.copy()
    ts_col = _pick_timestamp_col(df)
    if ts_col is None:
        return None, None
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        return None, None
    return df[ts_col].min(), df[ts_col].max()

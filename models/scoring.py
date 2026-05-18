import math
import numpy as np
import pandas as pd

from config import (
    SELF_MAP,
    Q8_GOOD,
    Q12_SCORE,
    Q11_R,
    CENTERS,
)


def clamp01(x: float) -> float:
    """
    将数值截断到 [0, 1] 区间。
    """
    return max(0.0, min(1.0, float(x)))


def score_A_q10(text: str) -> float:
    """
    根据 Q10 的自评节能意识，计算 A（行动/意识倾向）得分。
    映射后再线性归一化到 [0, 1]。
    """
    v = SELF_MAP.get(text)
    return 0.0 if v is None else (v - 1) / 3.0


def score_K1_q8(selected) -> float:
    """
    根据 Q8 多选题中命中的“正确节能做法”数量，计算 K1。
    """
    if not selected:
        return 0.0
    hit = sum(1 for x in selected if x in Q8_GOOD)
    return hit / len(Q8_GOOD)


def score_K2_q12(selected) -> float:
    """
    根据 Q12 判断题计算 K2。

    题目选项带有正负分，先求总分，再映射到 [0, 1]。
    如果未作答，返回 0.5 作为中性值。
    """
    if not selected:
        return 0.5

    chosen = set(selected)
    s = sum(val for t, val in Q12_SCORE.items() if t in chosen)

    pos = sum(1 for v in Q12_SCORE.values() if v > 0)
    neg = sum(1 for v in Q12_SCORE.values() if v < 0)
    min_s, max_s = -neg, pos

    return (s - min_s) / (max_s - min_s) if (max_s - min_s) > 0 else 0.0


def score_R_q11(text: str) -> float:
    """
    根据 Q11 主要影响因素，计算 R 分量。
    """
    return float(Q11_R.get(text, 0.0))


def compute_CS_from_4q(q8, q10, q11, q12):
    """
    由 4 道问卷题计算 CS（Cognition / Consciousness Score）。

    当前实现：
    - K = 0.5 * K1 + 0.5 * K2
    - A = score_A_q10(q10)
    - R = score_R_q11(q11)
    - CS = 0.4 * K + 0.3 * A + 0.3 * R

    返回
    ----
    cs : float
    parts : dict
        便于调试和页面展示的中间分项。
    """
    K1 = score_K1_q8(q8)
    K2 = score_K2_q12(q12)
    K = 0.5 * K1 + 0.5 * K2
    A = score_A_q10(q10)
    R = score_R_q11(q11)

    cs = 0.4 * K + 0.3 * A + 0.3 * R
    cs = clamp01(cs)

    parts = {
        "K1_q8": round(float(K1), 6),
        "K2_q12": round(float(K2), 6),
        "K": round(float(K), 6),
        "A_q10": round(float(A), 6),
        "R_q11": round(float(R), 6),
        "CS": round(float(cs), 6),
    }
    return cs, parts


def compute_BS_energy_week(dorm_out: pd.DataFrame, tau: float = 0.30) -> float:
    """
    基于最近一周“实际能耗相对基线的超出程度”计算 BS。

    逻辑：
    1. 取最近 7 天数据
    2. 用 diff = actual - baseline
    3. 只统计超出基线的部分（低于基线不扣分）
    4. 计算 over_ratio = sum(max(diff, 0)) / sum(baseline)
    5. 映射为 BS = 1 - over_ratio / tau，再截断到 [0, 1]

    含义：
    - 越接近基线或低于基线，BS 越高
    - 超出基线越多，BS 越低
    - tau 越小，惩罚越敏感
    """
    if dorm_out is None or dorm_out.empty:
        return 0.5

    df = dorm_out.copy()
    if "timestamp" not in df.columns:
        return 0.5

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if df.empty:
        return 0.5

    t_max = df["timestamp"].max()
    t_min = t_max - pd.Timedelta(days=7)
    wk = df[df["timestamp"] >= t_min].copy()

    if wk.empty:
        return 0.5

    actual_col = None
    if "energy_kwh" in wk.columns:
        actual_col = "energy_kwh"
    elif "kwh" in wk.columns:
        actual_col = "kwh"

    baseline_col = None
    if "baseline_pred" in wk.columns:
        baseline_col = "baseline_pred"
    elif "baseline_kwh" in wk.columns:
        baseline_col = "baseline_kwh"
    elif "yhat" in wk.columns:
        baseline_col = "yhat"

    if actual_col is None or baseline_col is None:
        return 0.5

    actual = pd.to_numeric(wk[actual_col], errors="coerce").fillna(0.0)
    baseline = pd.to_numeric(wk[baseline_col], errors="coerce").fillna(0.0)

    diff = actual - baseline
    over = diff.clip(lower=0)

    baseline_sum = float(baseline.sum())
    if baseline_sum <= 1e-9:
        return 0.5

    over_ratio = float(over.sum()) / baseline_sum
    bs = 1.0 - over_ratio / float(tau if tau > 1e-9 else 0.30)
    return clamp01(bs)


def compute_BS_energy_week_simple(actual_sum: float, baseline_sum: float, tau: float = 0.30) -> float:
    """
    简化版 BS 计算。
    适用于你已经提前聚合好 actual_sum 和 baseline_sum 的场景。
    """
    actual_sum = float(actual_sum)
    baseline_sum = float(baseline_sum)

    if baseline_sum <= 1e-9:
        return 0.5

    over_ratio = max(actual_sum - baseline_sum, 0.0) / baseline_sum
    bs = 1.0 - over_ratio / float(tau if tau > 1e-9 else 0.30)
    return clamp01(bs)


def classify_cluster(CS: float, BS: float):
    """
    根据 (CS, BS) 到各 cluster 中心的欧氏距离平方，分配最近类别。

    返回
    ----
    best_label : str
        A / B / C / D / E
    best_dist2 : float
        到最近中心的距离平方，便于调试或展示。
    """
    cs = float(CS)
    bs = float(BS)

    best_label = None
    best_dist2 = None

    for label, (cx, cy) in CENTERS.items():
        d2 = (cs - float(cx)) ** 2 + (bs - float(cy)) ** 2
        if best_dist2 is None or d2 < best_dist2:
            best_label = label
            best_dist2 = d2

    return best_label, float(best_dist2)

from __future__ import annotations

from typing import Dict, Optional, Tuple


def _clip01(x: float, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def compute_gap_index(cs: float, bs: float) -> float:
    """
    工程版 GI：直接使用 CS - BS。
    说明：
    - 不改变你现有 CS / BS 的定义；
    - 仅增加一个可解释辅助指标；
    - 数值范围通常落在 [-1, 1]，业务上更常见是 [-0.5, 0.8] 左右。
    """
    return round(_clip01(cs) - _clip01(bs), 6)



def label_awareness_level(cs: float) -> str:
    cs = _clip01(cs)
    if cs >= 0.80:
        return "high"
    if cs >= 0.55:
        return "medium"
    return "low"



def label_behavior_level(bs: float) -> str:
    bs = _clip01(bs)
    if bs >= 0.70:
        return "high"
    if bs >= 0.45:
        return "medium"
    return "low"



def label_gap_level(gi: float) -> str:
    try:
        g = float(gi)
    except Exception:
        g = 0.0
    if g >= 0.30:
        return "high"
    if g >= 0.12:
        return "medium"
    return "low"



def build_cluster_reason(
    cs: float,
    bs: float,
    gi: float,
    cluster_type: str,
    awareness_level: Optional[str] = None,
    behavior_level: Optional[str] = None,
    gap_level: Optional[str] = None,
    source: str = "rule_v2",
) -> str:
    awareness_level = awareness_level or label_awareness_level(cs)
    behavior_level = behavior_level or label_behavior_level(bs)
    gap_level = gap_level or label_gap_level(gi)

    cluster_desc = {
        "A": "高意识-低行为，差距显著",
        "B": "中高意识-低行为，需要强化执行",
        "C": "高意识-中行为，仍有提升空间",
        "D": "中等意识-中等行为，整体较平稳",
        "E": "高一致性或过渡型群体",
    }.get(str(cluster_type), "未定义群体")

    return (
        f"CS={float(cs):.3f}({awareness_level})，"
        f"BS={float(bs):.3f}({behavior_level})，"
        f"GI={float(gi):.3f}({gap_level})；"
        f"判为 {cluster_type} 类：{cluster_desc}。"
        f"来源={source}"
    )



def enrich_score_record(
    score: Dict,
    cluster_type: Optional[str] = None,
    cluster_dist2: Optional[float] = None,
    reason_source: str = "rule_v2",
) -> Dict:
    """
    给已有 score 记录补充 GI / level / reason 等字段。
    不改变原有核心字段，只做增量增强。
    """
    rec = dict(score or {})
    cs = _clip01(rec.get("CS", 0.5))
    bs = _clip01(rec.get("BS_energy_week", rec.get("BS", 0.5)))
    gi = compute_gap_index(cs, bs)
    awareness_level = label_awareness_level(cs)
    behavior_level = label_behavior_level(bs)
    gap_level = label_gap_level(gi)

    if cluster_type is not None:
        rec["cluster_type"] = cluster_type
    if cluster_dist2 is not None:
        rec["cluster_dist2"] = None if cluster_dist2 is None else round(float(cluster_dist2), 6)

    rec["GI"] = round(float(gi), 6)
    rec["awareness_level"] = awareness_level
    rec["behavior_level"] = behavior_level
    rec["gap_level"] = gap_level
    rec["cluster_reason"] = build_cluster_reason(
        cs=cs,
        bs=bs,
        gi=gi,
        cluster_type=rec.get("cluster_type", "E"),
        awareness_level=awareness_level,
        behavior_level=behavior_level,
        gap_level=gap_level,
        source=reason_source,
    )
    return rec


def classify_cluster_v2_from_fixed_scores(
    cs: float,
    bs: float,
    fallback_cluster_type: str,
    fallback_dist2: Optional[float] = None,
) -> Tuple[str, Optional[float]]:
    cs = _clip01(cs)
    bs = _clip01(bs)
    gi = compute_gap_index(cs, bs)
    base = str(fallback_cluster_type or "E")

    if cs >= 0.78 and bs <= 0.42 and gi >= 0.25:
        return "A", fallback_dist2

    if cs >= 0.58 and bs <= 0.46 and gi >= 0.12:
        return "B", fallback_dist2

    if cs >= 0.78 and 0.42 < bs < 0.72 and gi >= 0.08:
        return "C", fallback_dist2

    if (bs >= 0.72 and gi <= 0.12) or (cs >= 0.72 and bs >= 0.62 and gi <= 0.10):
        return "E", fallback_dist2

    if abs(cs - bs) <= 0.18:
        return "D", fallback_dist2

    if gi >= 0.18 and bs < 0.50:
        return "B", fallback_dist2

    if gi >= 0.10 and cs >= 0.72:
        return "C", fallback_dist2

    return "D", fallback_dist2


def classify_cluster_v2_from_fixed_scores(
    cs: float,
    bs: float,
    fallback_cluster_type=None,
    fallback_dist2=None,
):
    cs = _clip01(cs)
    bs = _clip01(bs)
    gi = compute_gap_index(cs, bs)

    # 保留旧逻辑的 fallback
    cluster_type = fallback_cluster_type or "E"
    dist2 = fallback_dist2

    # A: 高意识 + 低行为 + 高差距
    if cs >= 0.78 and bs <= 0.40 and gi >= 0.28:
        return "A", dist2

    # B: 中高意识 + 低行为 + 中高差距
    if cs >= 0.58 and bs <= 0.46 and gi >= 0.14:
        return "B", dist2

    # C: 高意识 + 中行为 + 仍有明显差距
    if cs >= 0.78 and 0.40 < bs < 0.72 and gi >= 0.10:
        return "C", dist2

    # E: 高一致性 / 高行为 / 差距小
    if (bs >= 0.72 and gi <= 0.18) or (cs >= 0.72 and bs >= 0.62 and gi <= 0.12):
        return "E", dist2

    # D: 中间协调型
    if cs >= 0.45 and bs >= 0.42 and gi < 0.14:
        return "D", dist2

    # 最后兜底：根据更接近哪种语义落到 B/C/D/E
    if gi >= 0.18 and bs < 0.50:
        return "B", dist2
    if gi >= 0.10 and cs >= 0.72:
        return "C", dist2
    if gi < 0.10 and bs >= 0.55:
        return "E", dist2
    return "D", dist2


def classify_cluster_v2_from_pool_distribution(
    as_val: float,
    bs_val: float,
    thresholds: Dict,
):
    try:
        a = _clip01(as_val)
        b = _clip01(bs_val)
    except Exception:
        return "D", None

    a_low = float(thresholds["AS_p33"])
    a_high = float(thresholds["AS_p67"])
    b_low = float(thresholds["BS_p33"])
    b_high = float(thresholds["BS_p67"])
    a_mid = float(thresholds["AS_p50"])
    b_mid = float(thresholds["BS_p50"])

    gi = compute_gap_index(a, b)
    dist2 = (a - a_mid) ** 2 + (b - b_mid) ** 2

    # A：高意识 + 低行为 + 高差距
    if a >= a_high and b <= b_low and gi >= 0.18:
        return "A", dist2

    # B：中高意识 + 低行为
    if a >= a_mid and b <= b_low and gi >= 0.08:
        return "B", dist2

    # C：高意识 + 中行为，仍有差距
    if a >= a_mid and b_low < b < b_high and gi >= 0.05:
        return "C", dist2

    # E：真正的一致性较高 / 行为较高
    if (b >= b_high and gi <= 0.14) or (a >= a_high and b >= b_mid and gi <= 0.08):
        return "E", dist2

    # D：中间协调型（显著放宽）
    if abs(a - b) <= 0.14:
        return "D", dist2

    # 其余兜底优先给 D，而不是 E
    return "D", dist2
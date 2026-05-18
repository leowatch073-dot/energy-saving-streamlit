import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

# ---------- CS: 4题 -> K/A/R ----------
SELF_MAP = {
    "很强，并总能付诸行动": 4,
    "较强，但有时会向舒适度妥协": 3,
    "一般，方便省事更重要": 2,
    "较弱，很少关注": 1,
}

def score_attitude_A(q1_text: str) -> float:
    v = SELF_MAP.get(q1_text)
    if v is None:
        return 0.0
    return (v - 1) / 3.0  # 4->1, 1->0

def score_responsibility_R(likert_1_to_5: int) -> float:
    # 1~5 -> 0~1
    if likert_1_to_5 is None:
        return 0.0
    return clamp01((likert_1_to_5 - 1) / 4.0)

# K1：多选“好习惯识别”（你原 q8 的思想）
QK1_GOOD = {
    "设定26℃及以上（夏）/20℃及以下（冬）",
    "无人时或离开前必定关闭空调",
    "使用时会紧闭门窗",
    "会定期清洁空调滤网",
}
def score_knowledge_K1(selected: List[str]) -> float:
    if not selected:
        return 0.0
    hit = sum(1 for x in selected if x in QK1_GOOD)
    return hit / len(QK1_GOOD)  # 0~1

# K2：误区纠正（你原 q12 的思想：对+1，错-1，再归一化到0~1）
QK2_SCORE = {
    "夏季空调温度设定越低越舒适": -1,
    "长时间开空调时应关闭门窗以提高效率": 1,
    "空调运行时打开电扇辅助，能更快降温且更节能": 1,
    "定期清洁滤网对节能基本没影响": -1,
}
def score_knowledge_K2(selected: List[str]) -> float:
    if selected is None:
        selected = []
    # 中性策略：没选任何项 -> 0.5（你 JS 里的做法）
    if len(selected) == 0:
        return 0.5

    s = 0
    pos = sum(1 for v in QK2_SCORE.values() if v > 0)
    neg = sum(1 for v in QK2_SCORE.values() if v < 0)
    chosen = set(selected)
    for text, val in QK2_SCORE.items():
        if text in chosen:
            s += val

    min_s, max_s = -neg, pos
    return (s - min_s) / (max_s - min_s) if (max_s - min_s) > 0 else 0.0

def compute_CS_4q(ans: Dict[str, Any]) -> Dict[str, Any]:
    """
    ans 示例：
      {
        "q1_attitude": "较强，但有时会向舒适度妥协",
        "q2_responsibility": 4,   # 1~5
        "q3_knowledge_habits": [...],
        "q4_knowledge_beliefs": [...]
      }
    """
    A = score_attitude_A(ans.get("q1_attitude"))
    R = score_responsibility_R(ans.get("q2_responsibility"))
    K1 = score_knowledge_K1(ans.get("q3_knowledge_habits") or [])
    K2 = score_knowledge_K2(ans.get("q4_knowledge_beliefs") or [])
    K = (K1 + K2) / 2.0

    # 采用你中文摘取/需求计划书：CS = 0.4K + 0.3A + 0.3R
    CS = 0.4 * K + 0.3 * A + 0.3 * R

    return {
        "CS": round(CS, 6),
        "parts": {"K": round(K, 6), "A": round(A, 6), "R": round(R, 6), "K1": round(K1, 6), "K2": round(K2, 6)},
        "version": "CS_4q_KAR_v1",
    }

# ---------- BS: 4题 -> Temp/Habit/Practice/Mode ----------
def compute_BS_4q(ans: Dict[str, Any]) -> Dict[str, Any]:
    # 这里假设你已经把每题映射到 0~1（最省事）
    Temp = clamp01(ans.get("temp_score", 0.0))
    Habit = clamp01(ans.get("habit_score", 0.0))
    Practice = clamp01(ans.get("practice_score", 0.0))
    Mode = clamp01(ans.get("mode_score", 0.0))

    BS = 0.4 * Temp + 0.25 * Habit + 0.25 * Practice + 0.1 * Mode
    return {
        "BS": round(BS, 6),
        "parts": {"Temp": round(Temp, 6), "Habit": round(Habit, 6), "Practice": round(Practice, 6), "Mode": round(Mode, 6)},
        "version": "BS_4q_THPM_v1",
    }

# ---------- Cluster: nearest-center ----------
CENTERS = {
    "A": (0.83, 0.3129),
    "B": (0.97, 0.5705),
    "C": (0.56, 0.3745),
    "D": (0.71, 0.6501),
    "E": (0.96, 0.8129),
}

def classify_cluster(CS: float, BS: float) -> Dict[str, Any]:
    best = None
    best_d = None
    for k, (cs0, bs0) in CENTERS.items():
        d = (CS - cs0) ** 2 + (BS - bs0) ** 2
        if best_d is None or d < best_d:
            best, best_d = k, d
    return {"cluster_type": best, "distance2": round(float(best_d), 6), "centers_version": "paper_centers_v1"}
import math
import numpy as np
import pandas as pd


def linucb_init(alpha=0.8):
    """
    初始化 LinUCB 参数容器。

    返回结构：
    {
        "alpha": float,
        "arms": {
            arm_id: {
                "A": np.ndarray,
                "b": np.ndarray,
                "select_count": int,
                "reward_sum": float,
            }
        }
    }

    说明：
    - A 为 d×d 矩阵，初始为单位阵
    - b 为 d 维向量，初始为 0
    - 当前默认状态维度 d=4，对应：
      [CS, BS_energy_week, recent_diff_3h, is_weekend]
    """
    return {
        "alpha": float(alpha),
        "arms": {}
    }


def ensure_arm(params: dict, arm_id: str, d: int = 4):
    """
    确保指定 arm 在 params 中存在。

    若不存在，则按 LinUCB 默认初值创建：
    - A = I
    - b = 0
    - select_count = 0
    - reward_sum = 0.0
    """
    if "arms" not in params:
        params["arms"] = {}

    if arm_id not in params["arms"]:
        params["arms"][arm_id] = {
            "A": np.eye(d, dtype=float),
            "b": np.zeros(d, dtype=float),
            "select_count": 0,
            "reward_sum": 0.0,
        }


def linucb_score(params: dict, arm_id: str, x):
    """
    计算某个 arm 在状态向量 x 下的 LinUCB 分数。

    公式：
    p = theta^T x + alpha * sqrt(x^T A^{-1} x)
    其中：
    - theta = A^{-1} b
    - 第一项为利用项（exploitation）
    - 第二项为探索项（exploration）

    返回
    ----
    detail : dict
        {
            "arm_id": ...,
            "score": ...,
            "mu": ...,
            "bonus": ...,
            "theta": [...],
            "x": [...]
        }
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    ensure_arm(params, arm_id, d=len(x))

    alpha = float(params.get("alpha", 0.8))
    arm = params["arms"][arm_id]

    A = np.asarray(arm["A"], dtype=float)
    b = np.asarray(arm["b"], dtype=float)

    A_inv = np.linalg.inv(A)
    theta = A_inv @ b

    mu = float(theta @ x)
    bonus = float(alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))
    score = mu + bonus

    return {
        "arm_id": arm_id,
        "score": float(score),
        "mu": float(mu),
        "bonus": float(bonus),
        "theta": theta.tolist(),
        "x": x.tolist(),
    }


def linucb_choose_arm(params: dict, candidate_arms, x):
    """
    从候选 arm 中选择当前 LinUCB 分数最高的 arm。

    参数
    ----
    params : dict
        LinUCB 参数容器
    candidate_arms : list[str]
        当前允许参与决策的 arm 列表
    x : array-like
        当前状态向量，默认顺序应为：
        [CS, BS_energy_week, recent_diff_3h, is_weekend]

    返回
    ----
    best_arm : str
        得分最高的 arm
    best_detail : dict
        对应该 arm 的完整打分详情
    details : list[dict]
        所有候选 arm 的打分结果，按 score 从高到低排序
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if not candidate_arms:
        raise ValueError("candidate_arms 为空，无法进行 LinUCB 选择。")

    details = []
    for arm_id in candidate_arms:
        detail = linucb_score(params, arm_id, x)
        details.append(detail)

    details = sorted(details, key=lambda z: z["score"], reverse=True)
    best_detail = details[0]
    best_arm = best_detail["arm_id"]

    return best_arm, best_detail, details


def linucb_update(params: dict, arm_id: str, x, reward: float):
    """
    使用观测到的 reward 对指定 arm 做一次 LinUCB 更新。

    更新公式：
    - A <- A + x x^T
    - b <- b + r x

    说明：
    - 这是原地更新（in-place）
    - 当前函数不会新建 params 副本
    - reward 同时累计到 reward_sum
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    r = float(reward)

    ensure_arm(params, arm_id, d=len(x))
    arm = params["arms"][arm_id]

    A = np.asarray(arm["A"], dtype=float)
    b = np.asarray(arm["b"], dtype=float)

    A = A + np.outer(x, x)
    b = b + r * x

    arm["A"] = A
    arm["b"] = b
    arm["select_count"] = int(arm.get("select_count", 0)) + 1
    arm["reward_sum"] = float(arm.get("reward_sum", 0.0)) + r


def parse_state_vector(CS, BS_energy_week, recent_diff_3h, is_weekend):
    """
    构造 LinUCB 使用的标准状态向量。

    固定顺序：
    [CS, BS_energy_week, recent_diff_3h, is_weekend]

    注意：
    这个顺序训练、打分、更新时必须保持一致。
    """
    return np.array(
        [
            float(CS),
            float(BS_energy_week),
            float(recent_diff_3h),
            float(is_weekend),
        ],
        dtype=float,
    )


def build_rl_monitor_df(params: dict) -> pd.DataFrame:
    """
    将当前 LinUCB 参数整理成便于监控展示的表格。

    输出字段包括：
    - arm_id
    - select_count
    - reward_sum
    - theta_0 ~ theta_{d-1}

    适用于 Admin 页的调试与解释面板。
    """
    rows = []

    for arm_id, arm in params.get("arms", {}).items():
        A = np.asarray(arm["A"], dtype=float)
        b = np.asarray(arm["b"], dtype=float)

        try:
            theta = np.linalg.inv(A) @ b
        except Exception:
            theta = np.full(len(b), np.nan)

        row = {
            "arm_id": arm_id,
            "select_count": int(arm.get("select_count", 0)),
            "reward_sum": float(arm.get("reward_sum", 0.0)),
        }

        for i, v in enumerate(theta):
            row[f"theta_{i}"] = float(v)

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["arm_id", "select_count", "reward_sum", "theta_0", "theta_1", "theta_2", "theta_3"]
        )

    return pd.DataFrame(rows).sort_values(["select_count", "arm_id"], ascending=[False, True]).reset_index(drop=True)
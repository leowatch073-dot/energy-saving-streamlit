from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


"""
strategy_library_1.py
---------------------
用于 EnergyApp 的消息策略文案库（第一版）。

当前设计目标：
1. 优先服务 Message 页面；
2. 先重点支持 P（消息提示）与 R（消息推送）两类；
3. F（数据反馈）和 S（社会比较）先保留占位模板，便于后续与 Home 页联动；
4. 提供统一的 arm_id -> 模板查找 / 消息生成函数。

建议命名规范：
- P_tip_xxx_xx         信息提示 / 小贴士
- R_push_xxx_xx        指示性提醒 / 推送消息
- F_feedback_xxx_xx    数据反馈（后续更适合接 Home 波动图）
- S_social_xxx_xx      社会比较（后续更适合接 Home 排行榜）
"""


# =========================
# 1) 文案模板库
# =========================
STRATEGY_LIBRARY: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------
    # P = Information Prompt / 消息提示 / 小贴士
    # -----------------------------------------------------------------
    "P_tip_temp_01": {
        "strategy_type": "P",
        "channel": "tip",
        "title": "温度设置小贴士",
        "body": "将空调设置在更合理的温度区间，通常更有利于兼顾舒适与节能，避免因设定过低而产生不必要耗电。",
        "goal": "知识提示",
        "tone": "gentle",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "知道了",
    },
    "P_tip_leave_01": {
        "strategy_type": "P",
        "channel": "tip",
        "title": "离室断电小贴士",
        "body": "短时间离开宿舍时也可能持续产生空调能耗。出门前顺手关闭空调，有助于减少不必要浪费。",
        "goal": "行为提示",
        "tone": "gentle",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "知道了",
    },
    "P_tip_window_01": {
        "strategy_type": "P",
        "channel": "tip",
        "title": "门窗状态小贴士",
        "body": "使用空调时保持门窗关闭，有助于减少冷量流失，使制冷效率更稳定。",
        "goal": "知识提示",
        "tone": "gentle",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "知道了",
    },
    "P_tip_fan_01": {
        "strategy_type": "P",
        "channel": "tip",
        "title": "配合风扇使用小贴士",
        "body": "在适当场景下配合风扇使用，可能帮助提升体感舒适度，并减少对低温设定的依赖。",
        "goal": "替代建议",
        "tone": "gentle",
        "suitable_clusters": ["B", "C", "E"],
        "cta_text": "知道了",
    },
    "P_tip_filter_01": {
        "strategy_type": "P",
        "channel": "tip",
        "title": "设备维护小贴士",
        "body": "保持空调滤网清洁有助于改善送风效率。设备状态更稳定时，也更容易实现节能运行。",
        "goal": "设备维护提示",
        "tone": "gentle",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "知道了",
    },

    # -----------------------------------------------------------------
    # R = Regular Reminder / 消息推送 / 指示性消息
    # -----------------------------------------------------------------
    "R_push_leave_01": {
        "strategy_type": "R",
        "channel": "push",
        "title": "离开宿舍前请确认空调状态",
        "body": "系统建议你在离开宿舍前检查空调是否仍在运行。若当前无需继续使用，请及时关闭。",
        "goal": "习惯养成",
        "tone": "directive",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "已完成",
    },
    "R_push_night_01": {
        "strategy_type": "R",
        "channel": "push",
        "title": "夜间使用提醒",
        "body": "若当前室温已较为稳定，可适当提高设定温度或缩短持续运行时间，以降低不必要能耗。",
        "goal": "夜间节能提醒",
        "tone": "directive",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "已完成",
    },
    "R_push_window_01": {
        "strategy_type": "R",
        "channel": "push",
        "title": "请检查门窗是否关闭",
        "body": "检测到当前更适合保持空调运行效率的条件。请确认门窗状态，减少冷量流失。",
        "goal": "即时行为提醒",
        "tone": "directive",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "已完成",
    },
    "R_push_setpoint_01": {
        "strategy_type": "R",
        "channel": "push",
        "title": "请检查空调设定温度",
        "body": "系统建议你查看当前空调设定值。若设定过低，可适度调整到更节能的区间。",
        "goal": "设定值提醒",
        "tone": "directive",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "已完成",
    },
    "R_push_habit_01": {
        "strategy_type": "R",
        "channel": "push",
        "title": "今日节能习惯提醒",
        "body": "请根据当前使用场景，优先采用更稳定的节能习惯，例如及时关机、避免过低设定温度或减少空转。",
        "goal": "习惯强化",
        "tone": "directive",
        "suitable_clusters": ["A", "B", "C", "D", "E"],
        "cta_text": "已完成",
    },

    # -----------------------------------------------------------------
    # F = Data Feedback / 数据反馈（当前先占位，后续建议在 Home 页主展示）
    # -----------------------------------------------------------------
    "F_feedback_weekly_01": {
        "strategy_type": "F",
        "channel": "feedback",
        "title": "本周能耗反馈",
        "body": "这里可用于展示本周能耗相对基线的变化结果，例如节能比例、变化趋势或阶段性反馈。",
        "goal": "结果反馈",
        "tone": "neutral",
        "suitable_clusters": ["C", "D", "E"],
        "cta_text": "查看详情",
    },
    "F_feedback_trend_01": {
        "strategy_type": "F",
        "channel": "feedback",
        "title": "近期能耗波动反馈",
        "body": "这里可用于衔接首页波动图，对近期能耗趋势进行解释，例如较前几日更平稳、较基线更接近目标区间等。",
        "goal": "趋势反馈",
        "tone": "neutral",
        "suitable_clusters": ["C", "D", "E"],
        "cta_text": "查看详情",
    },

    # -----------------------------------------------------------------
    # S = Social Comparison / 社会比较（当前先占位，后续建议在 Home 页主展示）
    # -----------------------------------------------------------------
    "S_social_rank_01": {
        "strategy_type": "S",
        "channel": "social",
        "title": "节能排行榜提示",
        "body": "这里可用于衔接首页排行榜模块，展示当前宿舍在同类宿舍中的相对节能位置。",
        "goal": "社会比较",
        "tone": "neutral",
        "suitable_clusters": ["A", "B", "D", "E"],
        "cta_text": "查看排行",
    },
    "S_social_peer_01": {
        "strategy_type": "S",
        "channel": "social",
        "title": "同伴表现提示",
        "body": "这里可用于展示相近宿舍或同组宿舍的节能表现，帮助用户形成参照与改进动力。",
        "goal": "同伴参照",
        "tone": "neutral",
        "suitable_clusters": ["A", "B", "D", "E"],
        "cta_text": "查看详情",
    },
}


# =========================
# 2) 群体 -> Message 页面推荐规划
# 这里只先强调 tip / push，适合当前 Message 页面
# =========================
CLUSTER_MESSAGE_PLAN: Dict[str, Dict[str, List[str]]] = {
    "A": {
        "tip": ["P_tip_leave_01", "P_tip_temp_01", "P_tip_window_01"],
        "push": ["R_push_leave_01", "R_push_setpoint_01", "R_push_window_01"],
    },
    "B": {
        "tip": ["P_tip_temp_01", "P_tip_fan_01", "P_tip_window_01"],
        "push": ["R_push_habit_01", "R_push_night_01", "R_push_setpoint_01"],
    },
    "C": {
        "tip": ["P_tip_fan_01", "P_tip_filter_01", "P_tip_leave_01"],
        "push": ["R_push_night_01", "R_push_leave_01", "R_push_habit_01"],
    },
    "D": {
        "tip": ["P_tip_window_01", "P_tip_filter_01", "P_tip_temp_01"],
        "push": ["R_push_window_01", "R_push_habit_01", "R_push_setpoint_01"],
    },
    "E": {
        "tip": ["P_tip_fan_01", "P_tip_filter_01", "P_tip_temp_01"],
        "push": ["R_push_night_01", "R_push_habit_01", "R_push_setpoint_01"],
    },
}


# =========================
# 3) 工具函数
# =========================
def get_template(arm_id: str) -> Optional[Dict[str, Any]]:
    """根据 arm_id 获取模板；若不存在则返回 None。"""
    if not arm_id:
        return None
    return STRATEGY_LIBRARY.get(str(arm_id))


def get_strategy_type(arm_id: str) -> str:
    tpl = get_template(arm_id)
    if tpl:
        return tpl.get("strategy_type", "Unknown")
    return "Unknown"


def get_channel(arm_id: str) -> str:
    tpl = get_template(arm_id)
    if tpl:
        return tpl.get("channel", "unknown")
    return "unknown"


def list_templates_by_type(strategy_type: str) -> List[str]:
    strategy_type = str(strategy_type).upper().strip()
    return [
        arm_id
        for arm_id, meta in STRATEGY_LIBRARY.items()
        if str(meta.get("strategy_type", "")).upper() == strategy_type
    ]


def list_templates_by_channel(channel: str) -> List[str]:
    channel = str(channel).strip().lower()
    return [
        arm_id
        for arm_id, meta in STRATEGY_LIBRARY.items()
        if str(meta.get("channel", "")).strip().lower() == channel
    ]


def get_cluster_message_plan(cluster_type: str) -> Dict[str, List[str]]:
    """获取某个 cluster 推荐的 tip/push 模板列表。"""
    if not cluster_type:
        return {"tip": [], "push": []}
    return CLUSTER_MESSAGE_PLAN.get(str(cluster_type).upper(), {"tip": [], "push": []})


def get_cluster_message_plan(cluster_type: str) -> Dict[str, List[str]]:
    """获取某个 cluster 推荐的 tip/push 模板列表。"""
    if not cluster_type:
        return {"tip": [], "push": []}
    return CLUSTER_MESSAGE_PLAN.get(str(cluster_type).upper(), {"tip": [], "push": []})


def normalize_cluster_type(cluster_type: str) -> str:
    """统一 cluster 表达，空值返回空字符串。"""
    return str(cluster_type or "").upper().strip()


def template_supports_cluster(arm_id: str, cluster_type: str) -> bool:
    """
    判断某条模板是否适用于指定 cluster。
    若模板未设置 suitable_clusters，则默认视为可用。
    若 cluster_type 为空，也默认可用。
    """
    tpl = get_template(arm_id)
    if tpl is None:
        return False

    cluster_type = normalize_cluster_type(cluster_type)
    if not cluster_type:
        return True

    supported = tpl.get("suitable_clusters", None)
    if not supported:
        return True

    supported_norm = [str(x).upper().strip() for x in supported]
    return cluster_type in supported_norm


def make_message_key(
    dorm_id: str,
    arm_id: str,
    cluster_type: str,
    ts: Optional[str] = None,
) -> str:
    """
    为消息生成一个稳定 key，便于 Home 写入 Messages 时去重。
    当前按 宿舍 + cluster + arm + 日期 粒度生成。
    """
    ts_text = _safe_now_str(ts)
    date_part = str(ts_text).split(" ")[0]
    dorm_part = str(dorm_id or "").strip()
    arm_part = str(arm_id or "").strip()
    cluster_part = normalize_cluster_type(cluster_type)
    return f"{dorm_part}__{cluster_part}__{arm_part}__{date_part}"



def _safe_now_str(ts: Optional[str] = None) -> str:
    return ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")




def build_message_from_arm(
    arm_id: str,
    dorm_id: str = "",
    cluster_type: str = "",
    ts: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    source_page: str = "home",
    score_basis: str = "simulated",
    validate_cluster: bool = True,
) -> Dict[str, Any]:
    """
    根据 arm_id 构建一条可直接写入 st.session_state['messages'] 的消息。

    新增字段：
    - template_id
    - message_key
    - source_page
    - score_basis
    - sync_status

    说明：
    - 当前阶段主要服务 Home -> Messages 的 P/R 同步
    - F/S 仍可通过本函数生成，但后续更适合在 Home 页面展示
    """
    cluster_type = normalize_cluster_type(cluster_type)
    ts_text = _safe_now_str(ts)
    tpl = get_template(arm_id)

    # arm 不存在
    if tpl is None:
        msg = {
            "title": f"未找到模板：{arm_id}",
            "body": "当前 arm_id 未在 strategy library 中注册，请检查命名是否一致。",
            "ts": ts_text,
            "dorm_id": dorm_id,
            "arm_id": arm_id,
            "template_id": arm_id,
            "cluster_type": cluster_type,
            "strategy_type": "Unknown",
            "channel": "unknown",
            "goal": "",
            "tone": "",
            "cta_text": "知道了",
            "read": False,
            "source_page": source_page,
            "score_basis": score_basis,
            "sync_status": "new",
            "message_key": make_message_key(
                dorm_id=dorm_id,
                arm_id=arm_id,
                cluster_type=cluster_type,
                ts=ts_text,
            ),
        }
        if overrides:
            msg.update(overrides)
        return msg

    # cluster 不匹配时，返回“受控提示消息”，便于排查
    if validate_cluster and not template_supports_cluster(arm_id, cluster_type):
        msg = {
            "title": f"模板与群体不匹配：{arm_id}",
            "body": f"当前模板 {arm_id} 不在 cluster {cluster_type} 的适用范围内，请检查分群与策略映射。",
            "ts": ts_text,
            "dorm_id": dorm_id,
            "arm_id": arm_id,
            "template_id": arm_id,
            "cluster_type": cluster_type,
            "strategy_type": tpl.get("strategy_type", "Unknown"),
            "channel": tpl.get("channel", "unknown"),
            "goal": tpl.get("goal", ""),
            "tone": tpl.get("tone", ""),
            "cta_text": tpl.get("cta_text", "知道了"),
            "read": False,
            "source_page": source_page,
            "score_basis": score_basis,
            "sync_status": "new",
            "message_key": make_message_key(
                dorm_id=dorm_id,
                arm_id=arm_id,
                cluster_type=cluster_type,
                ts=ts_text,
            ),
            "_template_warning": "cluster_mismatch",
        }
        if overrides:
            msg.update(overrides)
        return msg

    msg = {
        "title": tpl.get("title", "Untitled Message"),
        "body": tpl.get("body", ""),
        "ts": ts_text,
        "dorm_id": dorm_id,
        "arm_id": arm_id,
        "template_id": arm_id,
        "cluster_type": cluster_type,
        "strategy_type": tpl.get("strategy_type", "Unknown"),
        "channel": tpl.get("channel", "unknown"),
        "goal": tpl.get("goal", ""),
        "tone": tpl.get("tone", ""),
        "cta_text": tpl.get("cta_text", "知道了"),
        "read": False,
        "source_page": source_page,
        "score_basis": score_basis,
        "sync_status": "new",
        "message_key": make_message_key(
            dorm_id=dorm_id,
            arm_id=arm_id,
            cluster_type=cluster_type,
            ts=ts_text,
        ),
    }

    if overrides:
        msg.update(overrides)

    return msg


def build_tip_cards_for_cluster(
        cluster_type: str,
        dorm_id: str = "",
        ts: Optional[str] = None,
        source_page: str = "home",
        score_basis: str = "simulated",
    ) -> List[Dict[str, Any]]:

    """根据 cluster 生成 tip 类型的小贴士消息列表。"""
    plan = get_cluster_message_plan(cluster_type)
    return [
        build_message_from_arm(
            arm_id=arm_id,
            dorm_id=dorm_id,
            cluster_type=cluster_type,
            ts=ts,
            source_page=source_page,
            score_basis=score_basis,
        )
        for arm_id in plan.get("tip", [])
    ]

def build_push_cards_for_cluster(
    cluster_type: str,
    dorm_id: str = "",
    ts: Optional[str] = None,
    source_page: str = "home",
    score_basis: str = "simulated",
) -> List[Dict[str, Any]]:
    """根据 cluster 生成 push 类型的指示性消息列表。"""
    plan = get_cluster_message_plan(cluster_type)
    return [
        build_message_from_arm(
            arm_id=arm_id,
            dorm_id=dorm_id,
            cluster_type=cluster_type,
            ts=ts,
            source_page=source_page,
            score_basis=score_basis,
        )
        for arm_id in plan.get("push", [])
    ]


def get_primary_tip_for_cluster(
    cluster_type: str,
    dorm_id: str = "",
    ts: Optional[str] = None,
    source_page: str = "home",
    score_basis: str = "simulated",
) -> Optional[Dict[str, Any]]:
    tips = build_tip_cards_for_cluster(
        cluster_type=cluster_type,
        dorm_id=dorm_id,
        ts=ts,
        source_page=source_page,
        score_basis=score_basis,
    )
    return tips[0] if tips else None


def get_primary_push_for_cluster(
    cluster_type: str,
    dorm_id: str = "",
    ts: Optional[str] = None,
    source_page: str = "home",
    score_basis: str = "simulated",
) -> Optional[Dict[str, Any]]:
    pushes = build_push_cards_for_cluster(
        cluster_type=cluster_type,
        dorm_id=dorm_id,
        ts=ts,
        source_page=source_page,
        score_basis=score_basis,
    )
    return pushes[0] if pushes else None


def build_all_cards_for_cluster(
    cluster_type: str,
    dorm_id: str = "",
    ts: Optional[str] = None,
    source_page: str = "home",
    score_basis: str = "simulated",
) -> List[Dict[str, Any]]:
    """合并返回某个 cluster 的全部 Message 页推荐消息。"""
    return build_tip_cards_for_cluster(
        cluster_type,
        dorm_id=dorm_id,
        ts=ts,
        source_page=source_page,
        score_basis=score_basis,
    ) + build_push_cards_for_cluster(
        cluster_type,
        dorm_id=dorm_id,
        ts=ts,
        source_page=source_page,
        score_basis=score_basis,
    )

# =========================
# 4) 便于手动测试的小示例
# =========================
if __name__ == "__main__":
    sample = build_message_from_arm(
        arm_id="R_push_leave_01",
        dorm_id="Dorm-101",
        cluster_type="A",
    )
    print(sample)

    print("-" * 50)

    tips = build_tip_cards_for_cluster("B", dorm_id="Dorm-202")
    print(tips[0] if tips else "no tips")

    print("-" * 50)

    pushes = build_push_cards_for_cluster("C", dorm_id="Dorm-303")
    print(pushes[0] if pushes else "no pushes")